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

  // Global Dropdown Handler
  document.addEventListener("click", function (e) {
    const toggleBtn = e.target.closest(".dropdown-toggle");
    const allMenus = document.querySelectorAll(".dropdown-menu");

    if (toggleBtn) {
      e.preventDefault();
      e.stopPropagation();
      const dropdown = toggleBtn.closest(".dropdown");
      const menu = dropdown ? dropdown.querySelector(".dropdown-menu") : null;

      // Close all other dropdowns
      allMenus.forEach(function (m) {
        if (m !== menu) m.classList.remove("show");
      });

      if (menu) {
        menu.classList.toggle("show");
      }
    } else {
      // Click outside closes any open dropdown
      allMenus.forEach(function (m) {
        m.classList.remove("show");
      });
    }
  });
});

/**
 * Helper untuk pemanggilan API dengan penanganan format error standar
 */
async function apiRequest(url, options = {}) {
  try {
    options.headers = options.headers || {};
    const token = localStorage.getItem("access_token");
    if (token && !options.headers["Authorization"]) {
      options.headers["Authorization"] = `Bearer ${token}`;
    }
    if (options.body && typeof options.body === "string" && !options.headers["Content-Type"]) {
      options.headers["Content-Type"] = "application/json";
    }

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
