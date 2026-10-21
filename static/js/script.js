// Small check before the form is sent
document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("prediction-form");
  if (!form) return;

  form.addEventListener("submit", function (e) {
    const store = document.getElementById("store_location").value.trim().toLowerCase();
    const drop = document.getElementById("drop_location").value.trim().toLowerCase();

    if (store && drop && store === drop) {
      e.preventDefault();
      alert("Store and drop locations cannot be the same.");
    }
  });
});