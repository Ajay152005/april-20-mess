// dashboard.js

document.getElementById('logout-btn').addEventListener('click', function(){
    fetch('/logout', {
        method: 'POST'
    })
    .then(response => {
        if (response.ok) {
            window.location.href = '/login.html';
        } else {
            throw new Error('Logout failed');
        }
    })
    .catch(error => {
        alert(error.message);
    });
});