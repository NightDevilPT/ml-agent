"""
Docker Sandbox Utility for ML Training
=======================================

Provides a clean interface for:
- Creating isolated Docker containers for ML training
- Mounting dataset files and code into containers
- Executing training scripts with resource limits
- Extracting trained models back to host
- Automatic cleanup of containers and temporary files

Usage:
    from utils.sandbox import MlSandbox
    
    sandbox = MlSandbox()
    sandbox.prepare(
        dataset_files=["data.csv"], 
        code_string="import pandas...", 
        packages=["scikit-learn"]
    )
    result = sandbox.execute()
    if result.success:
        model_path = sandbox.extract_model()
    sandbox.cleanup()

    # Or use context manager (auto-cleanup)
    with MlSandbox() as sandbox:
        sandbox.prepare(dataset_files, code_string, packages)
        result = sandbox.execute()
        if result.success:
            model_path = sandbox.extract_model()
"""

from utils.logger import get_logger
import os
import tempfile
import shutil
import tarfile
import docker
from typing import List, Optional
from docker.models.containers import Container
from docker.errors import ImageNotFound, APIError


log = get_logger("sandbox")


class SandboxResult:
    """Result object returned by execute() method."""
    
    def __init__(self, success: bool, logs: str, error_message: str = ""):
        self.success = success
        self.logs = logs
        self.error_message = error_message
    
    def __repr__(self) -> str:
        return f"SandboxResult(success={self.success}, logs_length={len(self.logs)})"


class MlSandbox:
    """
    Docker sandbox for secure ML model training.
    
    Creates a temporary workspace, copies dataset files, generates train.py,
    runs training in isolated Docker container, and extracts the trained model.
    
    Can be used as a context manager:
        with MlSandbox() as sandbox:
            sandbox.prepare(...)
            result = sandbox.execute()
    """
    
    def __init__(
        self,
        image: str = None,
        memory_limit: str = None,
        timeout: int = None
    ):
        """
        Initialize sandbox with optional custom configuration.
        
        Args:
            image: Docker image to use (default: python:3.10-slim)
            memory_limit: Memory limit like "4g", "8g" (default: 8g)
            timeout: Command timeout in seconds (default: 600)
        """
        self.image = image or "python:3.10-slim"
        self.memory_limit = memory_limit or "8g"
        self.timeout = timeout or 600
        self.container_workspace = "/workspace"
        self.model_filename = "model.joblib"
        self.train_script_filename = "train.py"
        self.requirements_filename = "requirements.txt"
        self.install_command = "pip install --no-cache-dir"
        
        self.docker_client: Optional[docker.DockerClient] = None
        self.container: Optional[Container] = None
        self.workspace_dir: Optional[str] = None
        
        log.info("Sandbox initialized: image=%s, memory=%s, timeout=%ds", 
                 self.image, self.memory_limit, self.timeout)
    
    def prepare(
        self,
        dataset_files: List[str],
        code_string: str,
        packages: List[str] = None
    ) -> bool:
        """
        Prepare sandbox environment before execution.
        
        Creates temp directory, copies dataset files, writes train.py script.
        
        Args:
            dataset_files: List of paths to CSV files to mount
            code_string: Python code string for train.py
            packages: List of pip packages to install (optional)
        
        Returns:
            True if preparation successful, False otherwise
        """
        log.start("Preparing sandbox environment")
        
        try:
            # Create temporary workspace directory
            self.workspace_dir = tempfile.mkdtemp(prefix="ml_sandbox_")
            log.info("Created workspace: %s", self.workspace_dir)
            
            # Copy dataset files to workspace
            for file_path in dataset_files:
                if not os.path.exists(file_path):
                    log.error("Dataset file not found: %s", file_path)
                    return False
                
                dest_path = os.path.join(self.workspace_dir, os.path.basename(file_path))
                shutil.copy2(file_path, dest_path)
                log.info("Copied dataset: %s -> %s", file_path, dest_path)
            
            # Write train.py script
            train_script_path = os.path.join(self.workspace_dir, self.train_script_filename)
            with open(train_script_path, "w", encoding="utf-8") as f:
                f.write(code_string)
            log.info("Wrote %s (%d chars)", self.train_script_filename, len(code_string))
            
            # Write requirements.txt if packages provided
            if packages and len(packages) > 0:
                req_path = os.path.join(self.workspace_dir, self.requirements_filename)
                with open(req_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(packages))
                log.info("Wrote %s with %d packages", self.requirements_filename, len(packages))
            
            # Initialize Docker client
            self.docker_client = docker.from_env()
            
            # Pull image if not exists
            try:
                self.docker_client.images.get(self.image)
                log.info("Docker image found locally: %s", self.image)
            except ImageNotFound:
                log.info("Pulling Docker image: %s", self.image)
                self.docker_client.images.pull(self.image)
                log.info("Image pulled successfully")
            
            log.end("Sandbox preparation complete")
            return True
            
        except Exception as e:
            log.error("Preparation failed: %s", str(e))
            return False
    
    def execute(self) -> SandboxResult:
        """
        Execute training in Docker container.
        
        Creates container, installs packages (if requirements.txt exists),
        runs train.py, and captures output.
        
        Returns:
            SandboxResult with success flag and execution logs
        """
        log.start("Executing training in Docker container")
        
        if not self.workspace_dir:
            log.error("No workspace prepared. Call prepare() first.")
            return SandboxResult(False, "", "No workspace prepared")
        
        if not self.docker_client:
            log.error("Docker client not initialized.")
            return SandboxResult(False, "", "Docker client not initialized")
        
        try:
            # Build install command if requirements.txt exists
            req_file = os.path.join(self.workspace_dir, self.requirements_filename)
            install_cmd = ""
            if os.path.exists(req_file):
                install_cmd = f"{self.install_command} -r {self.requirements_filename} && "
                log.info("Package installation will be performed")
            
            # Full command to run
            full_command = f"bash -c '{install_cmd}python {self.container_workspace}/{self.train_script_filename}'"
            
            # Create and start container
            log.info("Creating container with memory limit: %s", self.memory_limit)
            self.container = self.docker_client.containers.run(
                image=self.image,
                command=full_command,
                volumes={self.workspace_dir: {"bind": self.container_workspace, "mode": "rw"}},
                working_dir=self.container_workspace,
                mem_limit=self.memory_limit,
                detach=True,
                remove=False
            )
            
            log.info("Container started: %s", self.container.id[:12])
            
            # Wait for completion with timeout
            try:
                result = self.container.wait(timeout=self.timeout)
                exit_code = result.get("StatusCode", -1)
                
                # Get logs
                logs = self.container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
                
                if exit_code == 0:
                    log.info("Training completed successfully (exit code 0)")
                    log.end("Execution successful")
                    return SandboxResult(True, logs, "")
                else:
                    log.error("Training failed with exit code: %d", exit_code)
                    # Extract last few lines of error for quick view
                    error_lines = logs.split('\n')[-5:] if logs else []
                    log.error("Last error lines: %s", '\n'.join(error_lines))
                    log.end("Execution failed")
                    return SandboxResult(False, logs, f"Exit code: {exit_code}")
                    
            except APIError as e:
                log.error("Container wait failed: %s", str(e))
                return SandboxResult(False, "", str(e))
                
        except APIError as e:
            log.error("Docker API error: %s", str(e))
            return SandboxResult(False, "", str(e))
        except Exception as e:
            log.error("Unexpected error: %s", str(e))
            return SandboxResult(False, "", str(e))
    
    def extract_model(self, model_filename: str = None) -> Optional[str]:
        """
        Extract trained model from container to host.
        
        Args:
            model_filename: Name of model file inside container (default: model.joblib)
        
        Returns:
            Path to extracted model on host, or None if extraction failed
        """
        log.start("Extracting model from container")
        
        if not self.container:
            log.error("No container available")
            return None
        
        if not self.workspace_dir:
            log.error("No workspace directory")
            return None
        
        filename = model_filename or self.model_filename
        container_model_path = os.path.join(self.container_workspace, filename)
        host_model_path = os.path.join(self.workspace_dir, filename)
        
        try:
            # Check if model file exists in container
            check_cmd = f"test -f {container_model_path} && echo 'exists' || echo 'not found'"
            exec_result = self.container.exec_run(check_cmd)
            if b"not found" in exec_result.output:
                log.error("Model file not found in container: %s", container_model_path)
                return None
            
            # Copy model from container to host workspace
            bits, _ = self.container.get_archive(container_model_path)
            
            # Write to temporary file then move (get_archive returns tar stream)
            temp_tar = os.path.join(self.workspace_dir, "model.tar")
            with open(temp_tar, "wb") as f:
                for chunk in bits:
                    f.write(chunk)
            
            # Extract tar
            with tarfile.open(temp_tar, "r") as tar:
                tar.extractall(self.workspace_dir)
            
            # Clean up tar
            os.remove(temp_tar)
            
            # Verify extracted model exists
            if os.path.exists(host_model_path):
                file_size = os.path.getsize(host_model_path)
                log.info("Model extracted to: %s (size: %d bytes)", host_model_path, file_size)
                log.end("Extraction successful")
                return host_model_path
            else:
                log.error("Model extraction failed - file not found after extraction")
                return None
                
        except Exception as e:
            log.error("Extraction failed: %s", str(e))
            return None
    
    def get_container_logs(self) -> str:
        """
        Get logs from container if still running or stopped.
        
        Returns:
            Container logs as string
        """
        if not self.container:
            return ""
        
        try:
            logs = self.container.logs(stdout=True, stderr=True).decode("utf-8", errors="replace")
            return logs
        except Exception as e:
            log.error("Failed to get logs: %s", str(e))
            return ""
    
    def cleanup(self) -> bool:
        """
        Remove container and delete temporary workspace.
        
        Always call this in finally block to prevent resource leaks.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        log.start("Cleaning up sandbox resources")
        success = True
        
        # Remove container
        if self.container:
            try:
                self.container.remove(force=True)
                log.info("Container removed: %s", self.container.id[:12])
            except Exception as e:
                log.error("Failed to remove container: %s", str(e))
                success = False
        
        # Remove workspace directory
        if self.workspace_dir and os.path.exists(self.workspace_dir):
            try:
                shutil.rmtree(self.workspace_dir)
                log.info("Workspace removed: %s", self.workspace_dir)
            except Exception as e:
                log.error("Failed to remove workspace: %s", str(e))
                success = False
        
        # Close Docker client
        if self.docker_client:
            try:
                self.docker_client.close()
                log.info("Docker client closed")
            except Exception as e:
                log.error("Failed to close Docker client: %s", str(e))
                success = False
        
        log.end("Cleanup complete (success=%s)", success)
        return success
    
    def __enter__(self):
        """Context manager entry - returns self for use in 'with' block."""
        log.info("Entering context manager")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup even if exception occurs."""
        log.info("Exiting context manager (cleaning up)")
        self.cleanup()
        # Don't suppress exceptions
        return False