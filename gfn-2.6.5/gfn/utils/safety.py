import time
import threading
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False
    print("[GFN:SAFETY] pynvml not installed. GPU monitoring disabled.")

class GPUMonitor:
    def __init__(self, threshold_temp=70, check_interval=5.0):
        self.threshold = threshold_temp
        self.interval = check_interval
        self.running = False
        self.paused_event = threading.Event()
        self.paused_event.set() # Initially not paused (set=True means go)
        
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                print(f"[GFN:SAFETY] GPU Monitor initialized. Threshold: {threshold_temp}C")
            except Exception as e:
                print(f"[GFN:SAFETY] Failed to init NVML: {e}")
                self.handle = None
        else:
            self.handle = None

    def start(self):
        if self.handle is None:
            return
        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def _monitor_loop(self):
        while self.running:
            try:
                temp = pynvml.nvmlDeviceGetTemperature(self.handle, pynvml.NVML_TEMPERATURE_GPU)
                if temp > self.threshold:
                    if self.paused_event.is_set():
                        print(f"\n[GFN:SAFETY] GPU Temp {temp}C > {self.threshold}C. Pausing training...")
                        self.paused_event.clear() # Block training
                else:
                    if not self.paused_event.is_set():
                        if temp < self.threshold - 5: # Hysteresis
                            print(f"\n[GFN:SAFETY] GPU Cooled to {temp}C. Resuming...")
                            self.paused_event.set()
                            
                time.sleep(self.interval)
            except Exception as e:
                print(f"[GFN:SAFETY] Error reading GPU temp: {e}")
                time.sleep(self.interval)

    def stop(self):
        self.running = False
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

    def check(self):
        """Called inside training loop to block if hot"""
        if self.handle is not None:
            self.paused_event.wait()
