/**
 * Puskesmas Sarang 1 - Main Application JavaScript
 * Mengelola interaksi visual, Chart.js, dan pemanggilan API asynchronous
 */

document.addEventListener("DOMContentLoaded", function () {
  // Auto-dismiss alerts after 5 seconds if marked
  const alerts = document.querySelectorAll(".alert-auto-dismiss");
  alerts.forEach(function (alert) {
    setTimeout(function () {
      alert.style.opacity = "0";
      setTimeout(function () { alert.remove(); }, 300);
    }, 5000);
  });
});

/**
 * Helper untuk pemanggilan API dengan penanganan format error standar
 */
async function apiRequest(url, options = {}) {
  try {
    const response = await fetch(url, options);
    const data = await response.json().catch(() => ({}));

    if (!response.ok) {
      const errorMsg = (data.error && data.error.message) || data.message || `HTTP Error ${response.status}`;
      const details = (data.error && data.error.details) || [];
      throw { message: errorMsg, details: details, status: response.status };
    }

    return data;
  } catch (err) {
    throw err;
  }
}
