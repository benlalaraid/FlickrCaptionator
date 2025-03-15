document.addEventListener('DOMContentLoaded', function() {
    // Initialize particles.js
    particlesJS('particles-js', {
        particles: {
            number: { value: 80, density: { enable: true, value_area: 800 } },
            color: { value: '#7b68ee' },
            shape: { type: 'circle' },
            opacity: { value: 0.5, random: true },
            size: { value: 3, random: true },
            line_linked: {
                enable: true,
                distance: 150,
                color: '#4a3aff',
                opacity: 0.4,
                width: 1
            },
            move: {
                enable: true,
                speed: 2,
                direction: 'none',
                random: true,
                straight: false,
                out_mode: 'out',
                bounce: false
            }
        },
        interactivity: {
            detect_on: 'canvas',
            events: {
                onhover: { enable: true, mode: 'grab' },
                onclick: { enable: true, mode: 'push' },
                resize: true
            },
            modes: {
                grab: { distance: 140, line_linked: { opacity: 1 } },
                push: { particles_nb: 4 }
            }
        },
        retina_detect: true
    });

    // Set current year in footer
    document.getElementById('current-year').textContent = new Date().getFullYear();

    // DOM Elements
    const fileInput = document.getElementById('file-input');
    const dropArea = document.getElementById('drop-area');
    const uploadContent = document.getElementById('upload-content');
    const previewContainer = document.getElementById('preview-container');
    const previewImage = document.getElementById('preview-image');
    const changeImageBtn = document.getElementById('change-image-btn');
    const generateBtn = document.getElementById('generate-btn');
    const resultContainer = document.getElementById('result-container');
    const captionText = document.getElementById('caption-text');
    const copyBtn = document.getElementById('copy-btn');
    const shareBtn = document.getElementById('share-btn');
    const loader = document.getElementById('loader');

    // Variables
    let selectedFile = null;

    // Event Listeners
    fileInput.addEventListener('change', handleFileSelect);
    changeImageBtn.addEventListener('click', resetImage);
    generateBtn.addEventListener('click', generateCaption);
    copyBtn.addEventListener('click', copyToClipboard);
    shareBtn.addEventListener('click', shareCaption);

    // Drag and drop functionality
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, highlight, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, unhighlight, false);
    });

    dropArea.addEventListener('drop', handleDrop, false);

    // Functions
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    function highlight() {
        dropArea.classList.add('highlight');
    }

    function unhighlight() {
        dropArea.classList.remove('highlight');
    }

    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length) {
            handleFiles(files);
        }
    }

    function handleFileSelect(e) {
        if (e.target.files.length) {
            handleFiles(e.target.files);
        }
    }

    function handleFiles(files) {
        if (files[0].type.startsWith('image/')) {
            selectedFile = files[0];
            displayPreview(selectedFile);
            generateBtn.disabled = false;
        } else {
            alert('Please select an image file');
        }
    }

    function displayPreview(file) {
        const reader = new FileReader();
        
        reader.onload = function(e) {
            previewImage.src = e.target.result;
            uploadContent.style.display = 'none';
            previewContainer.style.display = 'flex';
        }
        
        reader.readAsDataURL(file);
    }

    function resetImage() {
        uploadContent.style.display = 'flex';
        previewContainer.style.display = 'none';
        fileInput.value = '';
        selectedFile = null;
        generateBtn.disabled = true;
        resultContainer.style.display = 'none';
    }

    async function generateCaption() {
        if (!selectedFile) {
            alert('Please select an image first');
            return;
        }

        // Show loader
        loader.style.display = 'flex';
        
        try {
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            const response = await fetch('/generate-caption/', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error('Failed to generate caption');
            }
            
            const data = await response.json();
            
            // Display result
            captionText.textContent = data.caption;
            resultContainer.style.display = 'block';
            
            // Scroll to result
            resultContainer.scrollIntoView({ behavior: 'smooth' });
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred while generating the caption. Please try again.');
        } finally {
            // Hide loader
            loader.style.display = 'none';
        }
    }

    function copyToClipboard() {
        const text = captionText.textContent;
        navigator.clipboard.writeText(text)
            .then(() => {
                copyBtn.classList.add('copied');
                setTimeout(() => {
                    copyBtn.classList.remove('copied');
                }, 2000);
            })
            .catch(err => {
                console.error('Failed to copy text: ', err);
            });
    }

    function shareCaption() {
        const text = captionText.textContent;
        
        if (navigator.share) {
            navigator.share({
                title: 'AI Generated Caption',
                text: text
            })
            .catch(err => {
                console.error('Error sharing: ', err);
            });
        } else {
            // Fallback for browsers that don't support Web Share API
            copyToClipboard();
            alert('Caption copied to clipboard for sharing!');
        }
    }

    // Add some animation when scrolling
    const animateOnScroll = () => {
        const elements = document.querySelectorAll('.upload-container, .result-container');
        
        elements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top;
            const screenPosition = window.innerHeight / 1.3;
            
            if (elementPosition < screenPosition) {
                element.style.opacity = '1';
                element.style.transform = 'translateY(0)';
            }
        });
    };

    window.addEventListener('scroll', animateOnScroll);
    
    // Initial animation call
    animateOnScroll();
});
