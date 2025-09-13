# Project Setup Guide

This README provides step-by-step instructions to set up and run the project, which consists of a backend and frontend application.

## Prerequisites

Before starting, ensure you have the following installed:
- **Python 3.7+** (for backend)
- **Node.js and npm** (for frontend)
- **Git** (for version control)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone 
cd 
```

### 2. Backend Setup

#### Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

#### Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

#### Run Backend Server

```bash
# Start the backend server (adjust command based on your backend framework)
python main.py
```

### 3. Frontend Setup

**Open a new terminal** and navigate to the project directory:

```bash
cd frontend
```

#### Install Frontend Dependencies

```bash
npm install
```

#### Run Frontend Development Server

```bash
npm run dev
```




