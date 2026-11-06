/**
 * CloudIntellMal Dashboard Configuration
 *
 * To point the dashboard to an external FastAPI backend (e.g., hosted on Render, Railway, or AWS):
 * window.API_BASE_URL = "https://your-backend-service.onrender.com/api";
 *
 * When left undefined or empty, the dashboard auto-detects:
 * - When loaded locally via a separate dev server (e.g. port 3000, 5500), defaults to http://127.0.0.1:8000/api
 * - When served via FastAPI (port 8000) or deployed to Vercel, defaults to the relative "/api"
 */
window.API_BASE_URL = window.API_BASE_URL || "";

