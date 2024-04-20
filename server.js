// server.js

const express = require('express');
const bodyParser = require('body-parser');
const mongoose = require('mongoose');
const session = require('express-session');
const Mongostore= require('connect-mongo')(session);

const app = express();

const PORT = process.env.PORT || 3000;

//Middleware
app.use(bodyParser.json());
app.use(session({
    secret: 'secret', //Change this to a random string in production
    resave: false,
    saveUninitialized: false,
    store: new Mongostore({mongooseConection: mongoose.connection }),
    cookie: { maxAge: 3600000} // Session expiration time (1 hour)

}))

// Connect to MongoDB
mongoose.connect('mongodb://localhost:27017/myapp', { useNewUrlParser: true, useUnifiedTopology: true});
const db = mongoose.connection;

db.once('open', () => {
    console.log('Connected to MongoDB');
});

const isAutheticated = (req, res, next) => {
    if(req.session.user) {
        next();
    } else {
        res.status(401).json({error: 'Unauthorized' });
    }
};



// User schema
const userSchema = new mongoose.Schema({
    username: String,
    email: String,
    password: String,
    age: Number
});

const User = mongoose.model('User', userSchema);

//API endpoints
app.post('/signup', (req, res) => {
    const newUser = new User({
        username: req.body.userame,
        email: req.body.email,
        password: req.body.password,
        age: req.body.age
    });

    newUser.save((err, user) => {
        if (err) {
          res.status(500).json({error: err.message});
        } else {
            res.status(201).json(user);
        }
});
});

app.post('/logout', (req, res) => 
{
    // Clear session or token to log out the user
    //Redirect to login page or send success response
    res.status(200).json({ message: 'Logout successful' });
});

app.post('/login', (res, res) => {
    const {email, password } = req.body;
    User.findOne({ email, password} , (err, user) => {
        if (err) {
            res.status(500).json({ error: err.message });
        } else if (!user) {
            res.status(401).json({error: 'Invalid credentials'});
        } else {
            res.status(200).json({message: 'Login successful', user });
        }
    });
});

app.post('/logout', (req, res) => {
    // Destroy session to log out user
    req.session.destroy(err => {
        if (err) {
            res.status(500).json({ error: 'Logout failed'});
        } else {
            res.status(200).json({ message: 'Logout successful '});
        }
    });
});

// Start server
app.listen(PORT, () => {
console.log(`Server is running on post ${PORT}`);
});