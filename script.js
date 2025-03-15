document.addEventListener("DOMContentLoaded", function () {
    const frame = document.getElementById("tutorial-frame");
    const algorithmDropdown = document.getElementById("algorithm-dropdown");

    algorithmDropdown.addEventListener("change", function () {
        frame.src = this.value || "about:blank"; // Load the selected notebook or reset if none
    });
});
