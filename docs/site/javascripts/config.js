
// Progress tracking
document.addEventListener('DOMContentLoaded', function() {
    // Count checked boxes
    const checkboxes = document.querySelectorAll('.task-list-control');
    const progressBar = document.querySelector('.progress-fill');
    
    function updateProgress() {
        const total = checkboxes.length;
        const checked = document.querySelectorAll('.task-list-control:checked').length;
        const percentage = (checked / total) * 100;
        
        if (progressBar) {
            progressBar.style.width = percentage + '%';
            progressBar.textContent = Math.round(percentage) + '%';
        }
        
        // Save to localStorage
        localStorage.setItem('python-prep-progress', percentage);
    }
    
    // Load saved progress
    const savedProgress = localStorage.getItem('python-prep-progress');
    if (savedProgress && progressBar) {
        progressBar.style.width = savedProgress + '%';
        progressBar.textContent = Math.round(savedProgress) + '%';
    }
    
    // Listen for changes
    checkboxes.forEach(checkbox => {
        checkbox.addEventListener('change', updateProgress);
    });
    
    updateProgress();
});

// Code copy enhancement
document.addEventListener('DOMContentLoaded', function() {
    const codeBlocks = document.querySelectorAll('pre code');
    codeBlocks.forEach(block => {
        block.addEventListener('click', function() {
            // Flash effect on copy
            block.style.background = '#e3f2fd';
            setTimeout(() => {
                block.style.background = '';
            }, 200);
        });
    });
});
