const BACKEND_URL = import.meta.env.VITE_BACKEND_URL ?? "http://localhost:8000";

export async function uploadSegmentation({ file, algorithms, webhook_url = null }) {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("algorithms", JSON.stringify(algorithms));
  if (webhook_url) {
    formData.append("webhook_url", webhook_url);
  }

  const res = await fetch(`${BACKEND_URL}/segmentation/upload`, {
    method: "POST",
    body: formData,
  });

  const text = await res.text();

  if (!res.ok) {
    throw new Error(
      `Upload failed (${res.status} ${res.statusText}): ${text.slice(0, 200)}`,
    );
  }

  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(`Upload returned invalid JSON: ${text.slice(0, 200)}`);
  }

  if (!data?.task_id) throw new Error("Upload response missing task_id");
  return data.task_id;
}

export async function fetchStatus(taskId) {
  const res = await fetch(`${BACKEND_URL}/segmentation/status/${taskId}`);
  const text = await res.text();

  if (!res.ok) {
    throw new Error(
      `Status failed (${res.status} ${res.statusText}): ${text.slice(0, 200)}`,
    );
  }

  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error(`Status returned invalid JSON: ${text.slice(0, 200)}`);
  }

  return data;
}

export function subscribeToTask(taskId, onMessage) {
  const eventSource = new EventSource(`${BACKEND_URL}/segmentation/stream/${taskId}`);
  
  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
      if (data.status === "completed" || data.status === "failed") {
        eventSource.close();
      }
    } catch (e) {
      console.error("Failed to parse SSE message:", e);
    }
  };
  
  eventSource.onerror = () => {
    eventSource.close();
  };
  
  return () => eventSource.close();
}
