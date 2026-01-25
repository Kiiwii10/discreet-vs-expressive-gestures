import { wsUrl } from "/shared/api.js";

export function watchState(onState, onTest) {
  const socket = new WebSocket(wsUrl("/ws"));
  socket.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.type === "state") onState?.(data);
      if (data.type === "test") onTest?.(data);
    } catch {
      // ignore
    }
  };
  socket.onclose = () => {
    setTimeout(() => watchState(onState, onTest), 1500);
  };
  return socket;
}

export function connectControlSocket(onOpen) {
  const socket = new WebSocket(wsUrl("/control"));
  socket.onopen = () => onOpen?.(socket);
  socket.onclose = () => {
    setTimeout(() => connectControlSocket(onOpen), 1500);
  };
  return socket;
}

