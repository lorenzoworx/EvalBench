import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import ReplayApp from "./ReplayApp";
import { replayMode } from "./mode";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    {replayMode ? <ReplayApp /> : <App />}
  </StrictMode>,
);

