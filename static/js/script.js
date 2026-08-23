// Basic client-side validation before form submission
document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("prediction-form");
  if (!form) return;

  form.addEventListener("submit", function (e) {
    const origin = document.getElementById("origin").value.trim();
    const destination = document.getElementById("destination").value.trim();

    if (origin && destination && origin.toLowerCase() === destination.toLowerCase()) {
      e.preventDefault();
      alert("Origin and destination cannot be the same city.");
    }
  });
});