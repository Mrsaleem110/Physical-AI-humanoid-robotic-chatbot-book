import React, { createContext, useContext, useState, useEffect } from 'react';
import { authAPI } from '../api/auth';

const UserContext = createContext();

export const useUser = () => {
  const context = useContext(UserContext);
  if (!context) {
    throw new Error('useUser must be used within a UserProvider');
  }
  return context;
};

export const UserProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  // Check for existing user session on component mount
  useEffect(() => {
    const checkAuthStatus = () => {
      const currentUser = authAPI.getCurrentUser();
      if (currentUser) {
        setUser(currentUser);
      }
      setIsLoading(false);
    };

    checkAuthStatus();
  }, []);

  const login = (userData) => {
    setUser(userData.user || userData);
    // Store both user and token if provided
    if (userData.token) {
      localStorage.setItem('authToken', userData.token);
    }
    localStorage.setItem('currentUser', JSON.stringify(userData.user || userData));
  };

  const logout = () => {
    authAPI.logout(); // This clears both user and token from localStorage
    setUser(null);
  };

  const updateUserProfile = (profileData) => {
    if (user) {
      const updatedUser = { ...user, ...profileData };
      setUser(updatedUser);
      localStorage.setItem('currentUser', JSON.stringify(updatedUser));
    }
  };

  const value = {
    user,
    isLoading,
    login,
    logout,
    updateUserProfile
  };

  return (
    <UserContext.Provider value={value}>
      {children}
    </UserContext.Provider>
  );
};