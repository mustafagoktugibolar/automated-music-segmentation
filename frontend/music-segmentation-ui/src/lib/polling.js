export function createPoller({
  intervalMs = 3000,
  timeoutMs = 10 * 60 * 1000,
} = {}) {
  let timer = null;
  let startedAt = 0;

  function start(fn) {
    stop();
    startedAt = Date.now();
    timer = setInterval(async () => {
      if (Date.now() - startedAt > timeoutMs) return; // caller decides what to do
      await fn();
    }, intervalMs);
  }

  function stop() {
    if (timer) clearInterval(timer);
    timer = null;
  }

  function isRunning() {
    return timer != null;
  }

  function elapsedMs() {
    return startedAt ? Date.now() - startedAt : 0;
  }

  return { start, stop, isRunning, elapsedMs, timeoutMs };
}
