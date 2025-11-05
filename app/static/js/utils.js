/**
 * Shows a toast notification
 * @param {string} message - The message to display
 * @param {string} type - The type of toast (success, error, warning, info)
 * @param {number} duration - Duration in milliseconds (default: 3000)
 */
export function showToast(message, type = 'info', duration = 3000) {
    const toastContainer = document.getElementById('toast-container');
    if (!toastContainer) return;
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    
    // Add the toast to the container
    toastContainer.appendChild(toast);
    
    // Trigger the animation
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);
    
    // Remove the toast after the duration
    setTimeout(() => {
        toast.classList.remove('show');
        toast.addEventListener('transitionend', () => {
            toast.remove();
        }, { once: true });
    }, duration);
}

// Add CSS for the toast if it doesn't exist
if (!document.getElementById('toast-styles')) {
    const style = document.createElement('style');
    style.id = 'toast-styles';
    style.textContent = `
        .toast-container {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            max-width: 350px;
        }
        
        .toast {
            background: #333;
            color: white;
            padding: 12px 16px;
            border-radius: 4px;
            margin-bottom: 10px;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .toast.show {
            opacity: 1;
            transform: translateX(0);
        }
        
        .toast-success {
            background: #4caf50;
        }
        
        .toast-error {
            background: #f44336;
        }
        
        .toast-warning {
            background: #ff9800;
            color: #000;
        }
        
        .toast-info {
            background: #2196f3;
        }
        
        /* Dark mode adjustments */
        .dark-mode .toast {
            background: #424242;
            color: white;
        }
    `;
    document.head.appendChild(style);
}
