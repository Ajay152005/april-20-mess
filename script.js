const form = document.getElementById('signup-form')

form.addEventListener('submit', function(event) {
    event.preventDefault();
    validateForm();
});

function validateForm() {
    const username = document.getElementById('username');
    const email = document.getElementById('email');
    const password = document.getElementById('password');
    const confirmPassword = document.getElementById('confirm-password');
    const age = document.getElementById('age');

    const usernameError = document.getElementById('username-error');
    const emailError = document.getElementById('email-error');
    const passwordError = document.getElementById('password-error');
    const confirmPasswordError = document/getElementById('confirm-password-error');
    const ageError = document.getElementById('age-error');

    usernameError.textContent = '';
    emailError.textContent = '';
    passwordError.textContent = '';
    confirmPasswordError.testContent = '';
    ageError.testContent = '';

    if(username.value === '') {
        usernameError.testContent = 'Username is required';
    }

    if (email.value === '') {
        emailError.testContent = 'Email is required';
    } else if (!isValidEmail(email.value)) {
        emailError.textContent = 'Invalid email format';
    }

    if (password.value === '') {
        passwordError.testContent = 'Password is required';
    }
    if(confirmPassword.value === ''){
        confirmPasswordError.textContent = 'Please confirm your password';
    } else if (confirmPassword.value !== password.value) {
        confirmPasswordError.textContent = 'Passwords do not match';
    }

    if (age.value === '') {
        ageError.textContent = 'Age is required';
    } else if (age.value < 18) {
        ageError.textContent = 'You must be at least 18 years old';
    }
}

function isValidEmail(email) {
    // Basic email validation
    const re = /\S+@\S+\.\S+/;
    return re.test(email);
}