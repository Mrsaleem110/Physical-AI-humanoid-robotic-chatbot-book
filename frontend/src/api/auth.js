// Mock API for authentication with JWT token support (in a real implementation, this would connect to your backend)
export const authAPI = {
  // Helper function to create a mock JWT token
  createMockToken(user) {
    const payload = {
      userId: user.id,
      email: user.email,
      name: user.name,
      exp: Math.floor(Date.now() / 1000) + (60 * 60 * 24), // 24 hours
      iat: Math.floor(Date.now() / 1000)
    };

    // In a real implementation, this would be a proper JWT with signature
    // For mock, we'll just base64 encode the payload
    const tokenPayload = btoa(JSON.stringify(payload));
    return `mock.jwt.header.${tokenPayload}.signature`;
  },

  // Mock signup function
  async signup(userData) {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 500));

    // Create a mock user object
    const mockUser = {
      id: Math.random().toString(36).substr(2, 9),
      email: userData.email,
      name: userData.name,
      createdAt: new Date().toISOString(),
      // Include background information
      softwareExperience: userData.softwareExperience,
      hardwareExperience: userData.hardwareExperience,
      programmingLanguages: userData.programmingLanguages,
      roboticsExperience: userData.roboticsExperience,
      primaryGoal: userData.primaryGoal,
      availableTime: userData.availableTime,
      preferredLearningStyle: userData.preferredLearningStyle,
    };

    // Create a mock JWT token
    const token = this.createMockToken(mockUser);

    // Store user and token in localStorage
    localStorage.setItem('currentUser', JSON.stringify(mockUser));
    localStorage.setItem('authToken', token);

    return {
      user: mockUser,
      token: token,
      success: true,
      message: 'Account created successfully'
    };
  },

  // Mock login function
  async login(credentials) {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 500));

    // In a real implementation, this would validate credentials against a database
    // For this mock, we'll just return a user if the email exists in localStorage
    const storedUser = localStorage.getItem('currentUser');
    if (storedUser) {
      const user = JSON.parse(storedUser);
      // In a real implementation, you would hash and compare passwords
      if (user.email === credentials.email) {
        // Create a new token for this login session
        const token = this.createMockToken(user);

        // Update the stored token
        localStorage.setItem('authToken', token);

        return {
          user,
          token,
          success: true,
          message: 'Login successful'
        };
      }
    }

    // If no user found, return an error
    throw new Error('Invalid email or password');
  },

  // Mock logout function
  logout() {
    localStorage.removeItem('currentUser');
    localStorage.removeItem('authToken');
  },

  // Get current user from localStorage
  getCurrentUser() {
    const storedUser = localStorage.getItem('currentUser');
    const token = localStorage.getItem('authToken');

    if (!storedUser || !token) {
      return null;
    }

    const user = JSON.parse(storedUser);

    // Verify token is not expired (for mock implementation)
    try {
      const tokenParts = token.split('.');
      if (tokenParts.length === 3) {
        const payload = JSON.parse(atob(tokenParts[1]));
        const currentTime = Math.floor(Date.now() / 1000);

        if (payload.exp && payload.exp < currentTime) {
          // Token is expired, remove it
          localStorage.removeItem('authToken');
          localStorage.removeItem('currentUser');
          return null;
        }
      }
    } catch (e) {
      console.error('Error parsing token:', e);
      return null;
    }

    return user;
  },

  // Get current auth token
  getAuthToken() {
    return localStorage.getItem('authToken');
  },

  // Check if user is authenticated
  isAuthenticated() {
    const user = this.getCurrentUser();
    return !!user;
  },

  // Update user profile
  async updateProfile(userId, profileData) {
    // Simulate API call delay
    await new Promise(resolve => setTimeout(resolve, 500));

    const storedUser = localStorage.getItem('currentUser');
    if (storedUser) {
      const user = JSON.parse(storedUser);
      if (user.id === userId) {
        const updatedUser = { ...user, ...profileData };
        localStorage.setItem('currentUser', JSON.stringify(updatedUser));
        return { user: updatedUser, success: true };
      }
    }

    throw new Error('User not found');
  }
};