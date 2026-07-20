// Copy to clipboard functionality
const copyBtn = document.getElementById('copy-btn');
if (copyBtn) {
    copyBtn.addEventListener('click', function() {
        const codeBlock = document.querySelector('code').innerText;
        navigator.clipboard.writeText(codeBlock).then(() => {
            copyBtn.innerText = 'Copied!';
            copyBtn.style.color = 'var(--accent-3)';
            setTimeout(() => {
                copyBtn.innerText = 'Copy';
                copyBtn.style.color = 'var(--text-main)';
            }, 2000);
        }).catch(err => {
            console.error('Failed to copy: ', err);
        });
    });
}

// Scroll Reveal Animation via Intersection Observer
function reveal() {
    var reveals = document.querySelectorAll(".reveal");
    for (var i = 0; i < reveals.length; i++) {
        var windowHeight = window.innerHeight;
        var elementTop = reveals[i].getBoundingClientRect().top;
        var elementVisible = 100;
        if (elementTop < windowHeight - elementVisible) {
            reveals[i].classList.add("active");
        }
    }
}
window.addEventListener("scroll", reveal);
// Trigger once on load
reveal();
