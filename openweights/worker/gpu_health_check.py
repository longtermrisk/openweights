import datetime
import json
import logging
import os
import platform
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import psutil
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUHealthCheck:
    @staticmethod
    def check_cuda_memory() -> Tuple[bool, Optional[str]]:
        """
        Check if CUDA memory can be accessed properly with detailed diagnostics.
        Returns: (is_healthy: bool, error_message: Optional[str])
        """
        try:
            # Log CUDA device information
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                logger.info(f"Number of CUDA devices: {device_count}")

                for i in range(device_count):
                    # Get detailed memory info for each device
                    free_mem, total_mem = torch.cuda.mem_get_info(i)
                    used_mem = total_mem - free_mem
                    logger.info(
                        f"GPU {i} - Total memory: {total_mem / 1e9:.2f}GB, "
                        f"Used: {used_mem / 1e9:.2f}GB, Free: {free_mem / 1e9:.2f}GB"
                    )

                    # Check memory fragmentation
                    torch.cuda.memory_summary(device=i)

                    # Check if memory is critically low
                    if free_mem / total_mem < 0.1:  # Less than 10% free
                        return (
                            False,
                            f"Critical: Low GPU memory on device {i} (Less than 10% free)",
                        )

                    # Try to allocate and free a small tensor to test memory access
                    try:
                        test_tensor = torch.ones((1024, 1024), device=f"cuda:{i}")
                        del test_tensor
                        torch.cuda.empty_cache()
                    except Exception as e:
                        return (
                            False,
                            f"Memory allocation test failed on GPU {i}: {str(e)}",
                        )

                return True, None
            else:
                # Check system GPU drivers
                try:
                    subprocess.run(["nvidia-smi"], capture_output=True, check=True)
                    return (
                        False,
                        "CUDA unavailable but NVIDIA drivers detected - possible driver/CUDA mismatch",
                    )
                except:
                    return False, "CUDA unavailable and no NVIDIA drivers detected"

        except RuntimeError as e:
            # Log system memory state for debugging
            vm = psutil.virtual_memory()
            logger.error(
                f"System memory - Total: {vm.total / 1e9:.2f}GB, Available: {vm.available / 1e9:.2f}GB"
            )
            return False, f"CUDA memory check failed: {str(e)}"

    @staticmethod
    def check_ecc_errors() -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Enhanced check for ECC errors with detailed diagnostics.
        Returns: (is_healthy: bool, error_message: Optional[str], ecc_info: Optional[Dict])
        """
        try:
            ecc_info = {}

            # Check if ECC is enabled
            cmd_ecc_mode = [
                "nvidia-smi",
                "--query-gpu=index,ecc.mode.current",
                "--format=csv,noheader,nounits",
            ]
            ecc_mode_result = subprocess.run(
                cmd_ecc_mode, capture_output=True, text=True, check=True
            )

            for line in ecc_mode_result.stdout.strip().split("\n"):
                gpu_idx, ecc_mode = line.strip().split(",")
                ecc_info[gpu_idx] = {"mode": ecc_mode.strip()}
                logger.info(f"GPU {gpu_idx} ECC Mode: {ecc_mode.strip()}")

            # Get detailed ECC error counts
            cmd_detailed = [
                "nvidia-smi",
                "--query-gpu=index,ecc.errors.corrected.volatile.device_memory,"
                "ecc.errors.corrected.volatile.register_file,"
                "ecc.errors.corrected.volatile.l1_cache,"
                "ecc.errors.corrected.volatile.l2_cache,"
                "ecc.errors.uncorrected.volatile.device_memory,"
                "ecc.errors.uncorrected.volatile.register_file,"
                "ecc.errors.uncorrected.volatile.l1_cache,"
                "ecc.errors.uncorrected.volatile.l2_cache",
                "--format=csv,noheader,nounits",
            ]

            result = subprocess.run(
                cmd_detailed, capture_output=True, text=True, check=True
            )

            for line in result.stdout.strip().split("\n"):
                logger.info(f"ECC line: {line}")
                logger.info(f"ECC line type: {type(line)}")
                values = line.strip().strip("[] ").split(",")
                values = [v.strip("[] ") for v in values]
                gpu_idx = values[0].strip()
                logger.info(f"ECC gpu_idx: {gpu_idx}")
                logger.info(f"ECC values: {values}")

                ecc_info[gpu_idx].update(
                    {
                        "corrected_errors": {
                            "device_memory": (
                                int(values[1]) if values[1] != "N/A" else 0
                            ),
                            "register_file": (
                                int(values[2]) if values[2] != "N/A" else 0
                            ),
                            "l1_cache": int(values[3]) if values[3] != "N/A" else 0,
                            "l2_cache": int(values[4]) if values[4] != "N/A" else 0,
                        },
                        "uncorrected_errors": {
                            "device_memory": (
                                int(values[5]) if values[5] != "N/A" else 0
                            ),
                            "register_file": (
                                int(values[6]) if values[6] != "N/A" else 0
                            ),
                            "l1_cache": int(values[7]) if values[7] != "N/A" else 0,
                            "l2_cache": int(values[8]) if values[8] != "N/A" else 0,
                        },
                    }
                )

                # Log detailed error counts
                logger.info(
                    f"GPU {gpu_idx} ECC Error Details: {json.dumps(ecc_info[gpu_idx], indent=2)}"
                )

                # Check for critical conditions
                total_uncorrected = sum(
                    ecc_info[gpu_idx]["uncorrected_errors"].values()
                )
                total_corrected = sum(ecc_info[gpu_idx]["corrected_errors"].values())

                if total_uncorrected > 0:
                    return (
                        False,
                        f"Critical: Uncorrected ECC errors detected on GPU {gpu_idx}",
                        ecc_info,
                    )

                if total_corrected > 1000:
                    return (
                        False,
                        f"Warning: High number of corrected ECC errors on GPU {gpu_idx}",
                        ecc_info,
                    )

            return True, None, ecc_info

        except subprocess.CalledProcessError as e:
            logger.warning(f"ECC check failed: {str(e)}")
            return True, "ECC error check not applicable or not supported", None
        except Exception as e:
            logger.error(f"Unexpected error in ECC check: {str(e)}")
            return False, f"Error checking ECC status: {str(e)}", None

    @staticmethod
    def get_nvidia_smi_info() -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Get detailed GPU information using nvidia-smi.
        Returns: (is_healthy: bool, error_message: Optional[str], gpu_info: Optional[Dict])
        """
        try:
            # Query using CSV format instead of JSON
            cmd = [
                "nvidia-smi",
                "--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used",
                "--format=csv,nounits,noheader",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            # Parse CSV output into a dictionary
            gpus_info = []
            lines = result.stdout.strip().split("\n")
            for line in lines:
                values = line.strip().split(",")
                gpu_info = {
                    "gpu": [
                        {
                            "timestamp": values[0].strip(),
                            "name": values[1].strip(),
                            "temperature.gpu": float(values[2]),
                            "utilization.gpu": float(values[3]),
                            "utilization.memory": float(values[4]),
                            "memory.total": float(values[5]),
                            "memory.free": float(values[6]),
                            "memory.used": float(values[7]),
                        }
                    ]
                }
                gpus_info.append(gpu_info)

                # Check for critical issues
                for gpu in gpu_info["gpu"]:
                    if gpu["temperature.gpu"] > 85:
                        return (
                            False,
                            f"GPU temperature too high: {gpu['temperature.gpu']}°C",
                            gpus_info,
                        )

                    if gpu["utilization.memory"] > 95:
                        return (
                            False,
                            f"Memory utilization too high: {gpu['utilization.memory']}%",
                            gpus_info,
                        )

            return True, None, gpus_info

        except subprocess.CalledProcessError as e:
            return False, f"nvidia-smi command failed: {str(e)}", None
        except Exception as e:
            return False, f"Unexpected error during nvidia-smi check: {str(e)}", None

    @staticmethod
    def run_basic_cuda_test() -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Enhanced CUDA functionality test with detailed diagnostics.
        Returns: (is_healthy: bool, error_message: Optional[str], test_results: Optional[Dict])
        """
        test_results = {}
        try:
            if not torch.cuda.is_available():
                return False, "CUDA is not available", None

            device_count = torch.cuda.device_count()
            logger.info(f"Testing {device_count} CUDA devices")

            for device_idx in range(device_count):
                test_results[device_idx] = {}
                torch.cuda.set_device(device_idx)

                # Log device properties
                props = torch.cuda.get_device_properties(device_idx)
                logger.info(f"Testing GPU {device_idx}: {props.name}")
                test_results[device_idx]["device_info"] = {
                    "name": props.name,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "total_memory": props.total_memory,
                }

                # Test 1: Basic tensor operations
                try:
                    start_mem = torch.cuda.memory_allocated()
                    x = torch.rand(1000, 1000, device=f"cuda:{device_idx}")
                    y = torch.matmul(x, x)
                    del x, y
                    torch.cuda.empty_cache()
                    end_mem = torch.cuda.memory_allocated()
                    test_results[device_idx]["basic_ops"] = "passed"
                    test_results[device_idx]["memory_leak"] = end_mem <= start_mem
                except Exception as e:
                    test_results[device_idx]["basic_ops"] = f"failed: {str(e)}"
                    return (
                        False,
                        f"Basic operations failed on GPU {device_idx}: {str(e)}",
                        test_results,
                    )

                # Test 2: Memory bandwidth
                try:
                    start_time = torch.cuda.Event(enable_timing=True)
                    end_time = torch.cuda.Event(enable_timing=True)

                    start_time.record()
                    large_tensor = torch.ones(
                        1024, 1024, 1024, device=f"cuda:{device_idx}"
                    )
                    large_tensor *= 2
                    end_time.record()

                    torch.cuda.synchronize()
                    test_results[device_idx]["memory_bandwidth"] = (
                        start_time.elapsed_time(end_time)
                    )
                    del large_tensor
                    torch.cuda.empty_cache()
                except Exception as e:
                    test_results[device_idx]["memory_bandwidth"] = f"failed: {str(e)}"
                    logger.warning(
                        f"Memory bandwidth test failed on GPU {device_idx}: {str(e)}"
                    )

                # Test 3: Compute capability check
                try:
                    if props.major < 3:
                        return (
                            False,
                            f"GPU {device_idx} has outdated compute capability {props.major}.{props.minor}",
                            test_results,
                        )
                except Exception as e:
                    logger.error(f"Failed to check compute capability: {str(e)}")

            return True, None, test_results

        except Exception as e:
            logger.error(f"CUDA test failed with error: {str(e)}")
            return False, f"CUDA test failed: {str(e)}", test_results

    @staticmethod
    def get_detailed_gpu_diagnostics() -> Dict:
        """
        Gather detailed diagnostics about all GPUs in the system.
        Returns a dictionary with comprehensive GPU information.
        """
        diagnostics = {
            "timestamp": datetime.datetime.now().isoformat(),
            "system_info": {
                "os": platform.platform(),
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_version": (
                    torch.version.cuda if hasattr(torch.version, "cuda") else None
                ),
                "cudnn_version": (
                    torch.backends.cudnn.version()
                    if torch.backends.cudnn.is_available()
                    else None
                ),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                ),
            },
            "gpus": {},
        }

        try:
            # Get nvidia-smi version
            nvidia_smi_version = subprocess.run(
                ["nvidia-smi", "--version"], capture_output=True, text=True
            ).stdout.strip()
            diagnostics["system_info"]["nvidia_smi_version"] = nvidia_smi_version
        except Exception as e:
            logger.warning(f"Could not get nvidia-smi version: {e}")
            diagnostics["system_info"]["nvidia_smi_version"] = "Not available"

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_info = {}
                try:
                    # Basic CUDA device properties
                    props = torch.cuda.get_device_properties(i)
                    gpu_info["device_properties"] = {
                        "name": props.name,
                        "compute_capability": f"{props.major}.{props.minor}",
                        "total_memory": props.total_memory,
                        "multi_processor_count": props.multi_processor_count,
                        "max_threads_per_block": props.max_threads_per_block,
                        "max_threads_per_multiprocessor": props.max_threads_per_multiprocessor,
                        "warp_size": props.warp_size,
                        "max_shared_memory_per_block": props.max_shared_memory_per_block,
                        "max_shared_memory_per_multiprocessor": props.max_shared_memory_per_multiprocessor,
                        "max_registry_size_per_block": props.max_registry_size_per_block,
                        "clock_rate": props.clock_rate,
                        "memory_clock_rate": props.memory_clock_rate,
                        "memory_bus_width": props.memory_bus_width,
                        "l2_cache_size": props.l2_cache_size,
                        "max_threads_per_sm": props.max_threads_per_multiprocessor,
                        "is_multi_gpu_board": props.is_multi_gpu_board,
                        "async_engine_count": props.async_engine_count,
                    }

                    # Current memory status
                    free_mem, total_mem = torch.cuda.mem_get_info(i)
                    used_mem = total_mem - free_mem
                    gpu_info["memory_status"] = {
                        "total_memory": total_mem,
                        "free_memory": free_mem,
                        "used_memory": used_mem,
                        "memory_allocated": torch.cuda.memory_allocated(i),
                        "memory_reserved": torch.cuda.memory_reserved(i),
                        "max_memory_allocated": torch.cuda.max_memory_allocated(i),
                        "memory_utilization": (used_mem / total_mem) * 100,
                    }

                    # Get detailed nvidia-smi information
                    try:
                        cmd = [
                            "nvidia-smi",
                            f"--id={i}",
                            "--query-gpu=temperature.gpu,temperature.memory,utilization.gpu,utilization.memory,power.draw,power.limit,enforced.power.limit,fan.speed,pstate,clocks.current.graphics,clocks.current.memory",
                            "--format=csv,noheader,nounits",
                        ]
                        result = subprocess.run(
                            cmd, capture_output=True, text=True, check=True
                        )
                        values = result.stdout.strip().split(",")

                        gpu_info["nvidia_smi"] = {
                            "temperature": {
                                "gpu": float(values[0]),
                                "memory": float(values[1]),
                            },
                            "utilization": {
                                "gpu": float(values[2]),
                                "memory": float(values[3]),
                            },
                            "power": {
                                "current_draw": float(values[4]),
                                "limit": float(values[5]),
                                "enforced_limit": float(values[6]),
                            },
                            "fan_speed": float(values[7]),
                            "performance_state": values[8].strip(),
                            "clocks": {
                                "graphics": float(values[9]),
                                "memory": float(values[10]),
                            },
                        }
                    except Exception as e:
                        logger.warning(
                            f"Could not get nvidia-smi info for GPU {i}: {e}"
                        )
                        gpu_info["nvidia_smi"] = "Not available"

                    # Get process information
                    try:
                        cmd = [
                            "nvidia-smi",
                            f"--id={i}",
                            "--query-compute-apps=pid,used_memory,name",
                            "--format=csv,noheader,nounits",
                        ]
                        result = subprocess.run(
                            cmd, capture_output=True, text=True, check=True
                        )
                        processes = []
                        for line in result.stdout.strip().split("\n"):
                            if line.strip():
                                pid, mem, name = line.strip().split(",")
                                processes.append(
                                    {
                                        "pid": int(pid),
                                        "used_memory": mem.strip(),
                                        "name": name.strip(),
                                    }
                                )
                        gpu_info["running_processes"] = processes
                    except Exception as e:
                        logger.warning(f"Could not get process info for GPU {i}: {e}")
                        gpu_info["running_processes"] = "Not available"

                except Exception as e:
                    logger.error(f"Error gathering information for GPU {i}: {e}")
                    gpu_info["error"] = str(e)

                diagnostics["gpus"][f"gpu_{i}"] = gpu_info

        return diagnostics

    @classmethod
    def check_gpu_health(cls) -> Tuple[bool, List[str], Dict]:
        """
        Enhanced GPU health check with detailed diagnostics.
        Returns: (is_healthy: bool, list_of_errors: List[str], diagnostic_info: Dict)
        """
        errors = []
        diagnostic_info = {
            "system_info": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_version": (
                    torch.version.cuda if hasattr(torch.version, "cuda") else None
                ),
            }
        }

        # Check CUDA memory
        mem_healthy, mem_error = cls.check_cuda_memory()
        diagnostic_info["cuda_memory"] = {"healthy": mem_healthy, "error": mem_error}
        if not mem_healthy:
            errors.append(mem_error)

        # Check nvidia-smi info
        smi_healthy, smi_error, gpu_info = cls.get_nvidia_smi_info()
        diagnostic_info["nvidia_smi"] = {
            "healthy": smi_healthy,
            "error": smi_error,
            "info": gpu_info,
        }
        if not smi_healthy:
            errors.append(smi_error)

        # Check for ECC errors
        ecc_healthy, ecc_error, ecc_info = cls.check_ecc_errors()
        diagnostic_info["ecc"] = {
            "healthy": ecc_healthy,
            "error": ecc_error,
            "info": ecc_info,
        }
        if not ecc_healthy:
            errors.append(ecc_error)

        # Run enhanced CUDA test
        cuda_healthy, cuda_error, cuda_test_results = cls.run_basic_cuda_test()
        diagnostic_info["cuda_test"] = {
            "healthy": cuda_healthy,
            "error": cuda_error,
            "results": cuda_test_results,
        }
        if not cuda_healthy:
            errors.append(cuda_error)

        # If any errors were detected, gather detailed GPU diagnostics
        if errors:
            logger.warning("GPU health check failed. Gathering detailed diagnostics...")
            detailed_diagnostics = cls.get_detailed_gpu_diagnostics()
            diagnostic_info["detailed_gpu_diagnostics"] = detailed_diagnostics

            # Log comprehensive error report
            logger.error("GPU Health Check Failed - Detailed Error Report:")
            logger.error(f"Number of errors detected: {len(errors)}")
            for i, error in enumerate(errors, 1):
                logger.error(f"Error {i}: {error}")

            logger.error("System Environment:")
            for key, value in detailed_diagnostics["system_info"].items():
                logger.error(f"  {key}: {value}")

            for gpu_id, gpu_info in detailed_diagnostics["gpus"].items():
                logger.error(f"\nDetailed information for {gpu_id}:")
                if "device_properties" in gpu_info:
                    logger.error("  Device Properties:")
                    for prop, value in gpu_info["device_properties"].items():
                        logger.error(f"    {prop}: {value}")

                if "memory_status" in gpu_info:
                    logger.error("  Memory Status:")
                    for stat, value in gpu_info["memory_status"].items():
                        logger.error(f"    {stat}: {value}")

                if "nvidia_smi" in gpu_info and isinstance(
                    gpu_info["nvidia_smi"], dict
                ):
                    logger.error("  Runtime Statistics:")
                    for category, stats in gpu_info["nvidia_smi"].items():
                        if isinstance(stats, dict):
                            logger.error(f"    {category}:")
                            for stat, value in stats.items():
                                logger.error(f"      {stat}: {value}")
                        else:
                            logger.error(f"    {category}: {stats}")

                if "running_processes" in gpu_info and isinstance(
                    gpu_info["running_processes"], list
                ):
                    logger.error("  Running Processes:")
                    for proc in gpu_info["running_processes"]:
                        logger.error(
                            f"    PID: {proc['pid']}, Memory: {proc['used_memory']}, Name: {proc['name']}"
                        )

        return len(errors) == 0, errors, diagnostic_info
