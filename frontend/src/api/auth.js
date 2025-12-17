// Mock API for authentication (in a real implementation, this would connect to your backend)
export const authAPI = {
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

    // Store user in localStorage
    localStorage.setItem('currentUser', JSON.stringify(mockUser));

    return { user: mockUser, success: true };
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
      if (user.email === credentials.email) {
        return { user, success: true };
      }
    }

    // If no user found, return an error
    throw new Error('Invalid email or password');
  },

  // Mock logout function
  logout() {
    localStorage.removeItem('currentUser');
  },

  // Get current user from localStorage
  getCurrentUser() {
    const storedUser = localStorage.getItem('currentUser');
    return storedUser ? JSON.parse(storedUser) : null;
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