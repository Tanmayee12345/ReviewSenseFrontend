// script.js

// MockAPI URL (replace this with your actual MockAPI URL)
const apiUrl = "https://yourmockapi.com/users";

// Show Signup Form
function showSignupForm() {
    document.getElementById('signup-form').style.display = 'block';
    document.getElementById('login-form').style.display = 'none';
}

// Show Login Form
function showLoginForm() {
    document.getElementById('login-form').style.display = 'block';
    document.getElementById('signup-form').style.display = 'none';
}

// Handle Signup Form Submission
document.getElementById('signupForm').addEventListener('submit', function(event) {
    event.preventDefault();
    
    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirm-password').value;

    if (password !== confirmPassword) {
        alert("Passwords do not match!");
        return;
    }

    // Create new user
    const newUser = { name, email, password };

    // Call MockAPI to create user
    fetch(apiUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(newUser),
    })
    .then(response => response.json())
    .then(data => {
        alert("Signup successful!");
        showLoginForm();  // Show login form after successful signup
    })
    .catch(error => {
        alert("Error signing up: " + error);
    });
});

// Handle Login Form Submission
document.getElementById('loginForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const email = document.getElementById('login-email').value;
    const password = document.getElementById('login-password').value;

    // Check if the entered email and password match any user
    fetch(apiUrl)
        .then(response => response.json())
        .then(users => {
            const user = users.find(u => u.email === email && u.password === password);

            if (user) {
                alert("Login successful!");
                window.location.href = "landing.html";  // Redirect to landing page
            } else {
                alert("Invalid email or password!");
            }
        })
        .catch(error => {
            alert("Error during login: " + error);
        });
});
