if (document.getElementById('file'))
    document.getElementById('file').addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file) {
            // Tampilkan preview gambar
            const reader = new FileReader();
            reader.onload = function(e) {
                const preview = document.getElementById('preview');
                preview.src = e.target.result;
                preview.style.display = 'block'; // Tampilkan gambar
            };
            reader.readAsDataURL(file);
        }
    });


window.onload = function() {
    // Get the current path (without query params or fragments)
    let currentPath = window.location.pathname;

    // Mapping each path to its corresponding link
    const navbarLinks = {
        '/dataset': document.getElementById('dataset-link'),
        '/preprocessing': document.getElementById('preprocessing-link'),
        '/modeling': document.getElementById('modeling-link'),
        '/': document.getElementById('prediction-link'),
        '/evaluation': document.getElementById('evaluation-link')
    };

    // Remove 'active' from all links initially
    Object.values(navbarLinks).forEach(link => link.classList.remove('active'));

    if (currentPath.includes('dataset')) currentPath = '/dataset'
    console.log(currentPath)

    // Add 'active' class to the current page link
    if (navbarLinks[currentPath]) {
        navbarLinks[currentPath].classList.add('active');
    }
};