import React, { useState, useEffect } from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import { authAPI } from '../api/auth';
import { useUser } from '../contexts/UserContext';
import { useHistory } from '@docusaurus/router';

const AuthComponent = () => {
  const { login, logout, updateUserProfile } = useUser();
  const history = useHistory();
  const { siteConfig } = useDocusaurusContext();
  const baseUrl = (siteConfig && siteConfig.baseUrl) ? siteConfig.baseUrl : '/';
  const [isLogin, setIsLogin] = useState(true);
  const [step, setStep] = useState(1); // 1: Auth, 2: Background questions
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    name: '',
    // Background questions
    softwareExperience: '',
    hardwareExperience: '',
    programmingLanguages: '',
    roboticsExperience: '',
    primaryGoal: '',
    availableTime: '',
    preferredLearningStyle: '',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // If URL contains ?step=2 or the user is already logged in but missing profile,
  // open the personalization/background questions automatically.
  useEffect(() => {
    try {
      if (typeof window !== 'undefined') {
        const params = new URLSearchParams(window.location.search);
        if (params.get('step') === '2') {
          setIsLogin(false);
          setStep(2);
        }

        // If a current user exists but appears to lack personalization, go to step 2
        const currentUser = authAPI.getCurrentUser();
        if (currentUser) {
          const needsPersonalization = !currentUser.preferredLearningStyle || !currentUser.softwareExperience;
          if (needsPersonalization) {
            setIsLogin(false);
            setStep(2);
          }
        }
      }
    } catch (e) {
      // ignore URL parsing errors
    }
  }, []);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleBackgroundChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const handleAuthSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      if (isLogin) {
        // Login
        const response = await authAPI.login({
          email: formData.email,
          password: formData.password
        });
        login(response); // Pass the full response which includes user and token
        // Redirect to homepage after successful login using window navigation for reliability
        setTimeout(() => {
          window.location.href = baseUrl;
        }, 1000); // 1 second delay to allow for UI feedback
      } else {
        // Signup - first create the account without background info
        const signupData = {
          email: formData.email,
          password: formData.password,
          name: formData.name,
        };

        const response = await authAPI.signup(signupData);
        // Ensure the user context is updated immediately with the new user data
        login(response); // Pass the full response which includes user and token

        // Move to background questions after successful signup
        setStep(2);
        // Optionally, we could redirect to the book directory after both steps are complete
        return;
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBackgroundSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');

    try {
      // Update the user profile with background information
      const currentUser = authAPI.getCurrentUser();
      if (currentUser) {
        const response = await authAPI.updateProfile(currentUser.id, {
          softwareExperience: formData.softwareExperience,
          hardwareExperience: formData.hardwareExperience,
          programmingLanguages: formData.programmingLanguages,
          roboticsExperience: formData.roboticsExperience,
          primaryGoal: formData.primaryGoal,
          availableTime: formData.availableTime,
          preferredLearningStyle: formData.preferredLearningStyle,
        });

        // Update the user context
        updateUserProfile(response.user);

        alert('Profile information saved! You can now customize content in chapters.');

        // Redirect to homepage after successful profile completion using window navigation for reliability
        setTimeout(() => {
          window.location.href = baseUrl;
        }, 1000); // 1 second delay to allow the alert to show

        // Reset form for potential future use
        setFormData(prev => ({
          ...prev,
          softwareExperience: '',
          hardwareExperience: '',
          programmingLanguages: '',
          roboticsExperience: '',
          primaryGoal: '',
          availableTime: '',
          preferredLearningStyle: '',
        }));
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderAuthForm = () => (
    <div className="auth-form-container" style={{
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      borderRadius: '12px',
      padding: '30px',
      margin: '20px 0',
      maxWidth: '420px',
      boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(255,255,255,0.2)'
    }}>
      <div style={{ textAlign: 'center', marginBottom: '25px' }}>
        <h2 style={{
          color: 'white',
          margin: '0 0 5px 0',
          fontSize: '24px',
          fontWeight: '600'
        }}>
          {isLogin ? 'Welcome Back' : 'Join Our Community'}
        </h2>
        <p style={{
          color: 'rgba(255,255,255,0.8)',
          margin: '0',
          fontSize: '14px'
        }}>
          {isLogin ? 'Sign in to continue your learning journey' : 'Create an account to personalize your experience'}
        </p>
      </div>

      <div style={{
        display: 'flex',
        justifyContent: 'center',
        marginBottom: '20px',
        background: 'rgba(255,255,255,0.1)',
        borderRadius: '50px',
        padding: '4px'
      }}>
        <button
          onClick={() => setIsLogin(true)}
          style={{
            backgroundColor: isLogin ? 'white' : 'transparent',
            color: isLogin ? '#667eea' : 'rgba(255,255,255,0.8)',
            border: 'none',
            padding: '10px 20px',
            cursor: 'pointer',
            borderRadius: '50px',
            fontWeight: '600',
            fontSize: '14px',
            transition: 'all 0.3s ease',
            flex: 1
          }}
        >
          Login
        </button>
        <button
          onClick={() => setIsLogin(false)}
          style={{
            backgroundColor: !isLogin ? 'white' : 'transparent',
            color: !isLogin ? '#667eea' : 'rgba(255,255,255,0.8)',
            border: 'none',
            padding: '10px 20px',
            cursor: 'pointer',
            borderRadius: '50px',
            fontWeight: '600',
            fontSize: '14px',
            transition: 'all 0.3s ease',
            flex: 1
          }}
        >
          Sign Up
        </button>
      </div>

      {error && (
        <div style={{
          backgroundColor: 'rgba(248, 215, 218, 0.9)',
          color: '#721c24',
          padding: '12px',
          borderRadius: '8px',
          marginBottom: '15px',
          border: '1px solid rgba(245, 198, 203, 0.3)',
          fontSize: '14px'
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      <form onSubmit={handleAuthSubmit} style={{ position: 'relative' }}>
        {!isLogin && (
          <div style={{ marginBottom: '15px' }}>
            <input
              type="text"
              name="name"
              placeholder="Full Name"
              value={formData.name}
              onChange={handleChange}
              style={{
                width: '100%',
                padding: '12px 15px',
                border: '1px solid rgba(255,255,255,0.3)',
                borderRadius: '8px',
                backgroundColor: 'rgba(255,255,255,0.95)',
                fontSize: '16px',
                color: '#333',
                transition: 'all 0.3s ease'
              }}
              required={!isLogin}
            />
          </div>
        )}
        <div style={{ marginBottom: '15px' }}>
          <input
            type="email"
            name="email"
            placeholder="Email Address"
            value={formData.email}
            onChange={handleChange}
            style={{
              width: '100%',
              padding: '12px 15px',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '8px',
              backgroundColor: 'rgba(255,255,255,0.95)',
              fontSize: '16px',
              color: '#333',
              transition: 'all 0.3s ease'
            }}
            required
          />
        </div>
        <div style={{ marginBottom: '20px' }}>
          <input
            type="password"
            name="password"
            placeholder="Password"
            value={formData.password}
            onChange={handleChange}
            style={{
              width: '100%',
              padding: '12px 15px',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '8px',
              backgroundColor: 'rgba(255,255,255,0.95)',
              fontSize: '16px',
              color: '#333',
              transition: 'all 0.3s ease'
            }}
            required
          />
        </div>
        <button
          type="submit"
          disabled={loading}
          style={{
            width: '100%',
            padding: '14px',
            backgroundColor: loading ? '#4a569d' : '#007cba',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '16px',
            fontWeight: '600',
            transition: 'all 0.3s ease',
            textTransform: 'uppercase',
            letterSpacing: '0.5px'
          }}
        >
          {loading ? (
            <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <span style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                backgroundColor: 'white',
                display: 'inline-block',
                marginRight: '8px',
                animation: 'loading 1s infinite'
              }}></span>
              Processing...
            </span>
          ) : (isLogin ? 'Sign In' : 'Create Account')}
        </button>
      </form>

      <style jsx>{`
        @keyframes loading {
          0%, 80%, 100% { transform: scale(0); }
          40% { transform: scale(1.0); }
        }
      `}</style>
    </div>
  );

  const renderBackgroundForm = () => (
    <div className="background-form-container" style={{
      background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
      borderRadius: '12px',
      padding: '30px',
      margin: '20px 0',
      maxWidth: '600px',
      boxShadow: '0 10px 25px rgba(0,0,0,0.1)',
      backdropFilter: 'blur(10px)',
      border: '1px solid rgba(255,255,255,0.2)'
    }}>
      <div style={{ textAlign: 'center', marginBottom: '25px' }}>
        <h2 style={{
          color: 'white',
          margin: '0 0 5px 0',
          fontSize: '24px',
          fontWeight: '600'
        }}>
          Personalize Your Learning
        </h2>
        <p style={{
          color: 'rgba(255,255,255,0.8)',
          margin: '0',
          fontSize: '14px'
        }}>
          Help us understand your background to customize your learning experience
        </p>
      </div>

      {error && (
        <div style={{
          backgroundColor: 'rgba(248, 215, 218, 0.9)',
          color: '#721c24',
          padding: '12px',
          borderRadius: '8px',
          marginBottom: '15px',
          border: '1px solid rgba(245, 198, 203, 0.3)',
          fontSize: '14px'
        }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      <form onSubmit={handleBackgroundSubmit} style={{ position: 'relative' }}>
        <div style={{ marginBottom: '15px' }}>
          <label style={{
            display: 'block',
            marginBottom: '8px',
            color: 'rgba(255,255,255,0.9)',
            fontWeight: '500',
            fontSize: '14px'
          }}>
            Software Experience Level:
          </label>
          <select
            name="softwareExperience"
            value={formData.softwareExperience}
            onChange={handleBackgroundChange}
            style={{
              width: '100%',
              padding: '12px 15px',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '8px',
              backgroundColor: 'rgba(255,255,255,0.95)',
              fontSize: '16px',
              color: '#333',
              transition: 'all 0.3s ease'
            }}
            required
          >
            <option value="">Select your level</option>
            <option value="beginner">Beginner</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
          </select>
        </div>

        <div style={{ marginBottom: '15px' }}>
          <label style={{
            display: 'block',
            marginBottom: '8px',
            color: 'rgba(255,255,255,0.9)',
            fontWeight: '500',
            fontSize: '14px'
          }}>
            Hardware Experience Level:
          </label>
          <select
            name="hardwareExperience"
            value={formData.hardwareExperience}
            onChange={handleBackgroundChange}
            style={{
              width: '100%',
              padding: '12px 15px',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '8px',
              backgroundColor: 'rgba(255,255,255,0.95)',
              fontSize: '16px',
              color: '#333',
              transition: 'all 0.3s ease'
            }}
            required
          >
            <option value="">Select your level</option>
            <option value="none">No Experience</option>
            <option value="basic">Basic</option>
            <option value="intermediate">Intermediate</option>
            <option value="advanced">Advanced</option>
          </select>
        </div>

        <div style={{ marginBottom: '15px' }}>
          <label style={{
            display: 'block',
            marginBottom: '8px',
            color: 'rgba(255,255,255,0.9)',
            fontWeight: '500',
            fontSize: '14px'
          }}>
            Programming Languages you're familiar with:
          </label>
          <input
            type="text"
            name="programmingLanguages"
            placeholder="e.g., Python, C++, JavaScript"
            value={formData.programmingLanguages}
            onChange={handleBackgroundChange}
            style={{
              width: '100%',
              padding: '12px 15px',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '8px',
              backgroundColor: 'rgba(255,255,255,0.95)',
              fontSize: '16px',
              color: '#333',
              transition: 'all 0.3s ease'
            }}
            required
          />
        </div>

        <div style={{ marginBottom: '15px' }}>
          <label style={{
            display: 'block',
            marginBottom: '8px',
            color: 'rgba(255,255,255,0.9)',
            fontWeight: '500',
            fontSize: '14px'
          }}>
            Robotics Experience:
          </label>
          <textarea
            name="roboticsExperience"
            placeholder="Describe your robotics experience..."
            value={formData.roboticsExperience}
            onChange={handleBackgroundChange}
            rows={3}
            style={{
              width: '100%',
              padding: '12px 15px',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '8px',
              backgroundColor: 'rgba(255,255,255,0.95)',
              fontSize: '16px',
              color: '#333',
              transition: 'all 0.3s ease',
              resize: 'vertical'
            }}
          />
        </div>

        <div style={{ marginBottom: '15px' }}>
          <label style={{
            display: 'block',
            marginBottom: '8px',
            color: 'rgba(255,255,255,0.9)',
            fontWeight: '500',
            fontSize: '14px'
          }}>
            Primary Goal:
          </label>
          <select
            name="primaryGoal"
            value={formData.primaryGoal}
            onChange={handleBackgroundChange}
            style={{
              width: '100%',
              padding: '12px 15px',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '8px',
              backgroundColor: 'rgba(255,255,255,0.95)',
              fontSize: '16px',
              color: '#333',
              transition: 'all 0.3s ease'
            }}
            required
          >
            <option value="">Select your goal</option>
            <option value="education">Educational Purposes</option>
            <option value="career">Career Advancement</option>
            <option value="research">Research</option>
            <option value="hobby">Personal Interest/Hobby</option>
            <option value="startup">Start a Business</option>
          </select>
        </div>

        <div style={{ marginBottom: '15px' }}>
          <label style={{
            display: 'block',
            marginBottom: '8px',
            color: 'rgba(255,255,255,0.9)',
            fontWeight: '500',
            fontSize: '14px'
          }}>
            How much time can you dedicate weekly?
          </label>
          <select
            name="availableTime"
            value={formData.availableTime}
            onChange={handleBackgroundChange}
            style={{
              width: '100%',
              padding: '12px 15px',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '8px',
              backgroundColor: 'rgba(255,255,255,0.95)',
              fontSize: '16px',
              color: '#333',
              transition: 'all 0.3s ease'
            }}
            required
          >
            <option value="">Select time commitment</option>
            <option value="less-5">Less than 5 hours</option>
            <option value="5-10">5-10 hours</option>
            <option value="10-15">10-15 hours</option>
            <option value="more-15">More than 15 hours</option>
          </select>
        </div>

        <div style={{ marginBottom: '20px' }}>
          <label style={{
            display: 'block',
            marginBottom: '8px',
            color: 'rgba(255,255,255,0.9)',
            fontWeight: '500',
            fontSize: '14px'
          }}>
            Preferred Learning Style:
          </label>
          <select
            name="preferredLearningStyle"
            value={formData.preferredLearningStyle}
            onChange={handleBackgroundChange}
            style={{
              width: '100%',
              padding: '12px 15px',
              border: '1px solid rgba(255,255,255,0.3)',
              borderRadius: '8px',
              backgroundColor: 'rgba(255,255,255,0.95)',
              fontSize: '16px',
              color: '#333',
              transition: 'all 0.3s ease'
            }}
            required
          >
            <option value="">Select your style</option>
            <option value="visual">Visual (Diagrams, Videos)</option>
            <option value="auditory">Auditory (Lectures, Discussions)</option>
            <option value="reading">Reading/Writing</option>
            <option value="kinesthetic">Hands-on/Practical</option>
            <option value="mixed">Mixed Approach</option>
          </select>
        </div>

        <button
          type="submit"
          disabled={loading}
          style={{
            width: '100%',
            padding: '14px',
            backgroundColor: loading ? '#218838' : '#28a745',
            color: 'white',
            border: 'none',
            borderRadius: '8px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '16px',
            fontWeight: '600',
            transition: 'all 0.3s ease',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
            marginBottom: '10px'
          }}
        >
          {loading ? (
            <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <span style={{
                width: '12px',
                height: '12px',
                borderRadius: '50%',
                backgroundColor: 'white',
                display: 'inline-block',
                marginRight: '8px',
                animation: 'loading 1s infinite'
              }}></span>
              Saving...
            </span>
          ) : 'Complete Profile'}
        </button>

        <button
          type="button"
          onClick={() => {
            // Redirect to homepage after completing profile using window navigation for reliability
            window.location.href = baseUrl;
          }}
          disabled={loading}
          style={{
            width: '100%',
            padding: '12px',
            backgroundColor: loading ? '#5a6268' : 'rgba(255,255,255,0.2)',
            color: loading ? 'rgba(255,255,255,0.6)' : 'white',
            border: '1px solid rgba(255,255,255,0.3)',
            borderRadius: '8px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontSize: '14px',
            fontWeight: '500',
            transition: 'all 0.3s ease'
          }}
        >
          Go to Homepage
        </button>
      </form>

      <style jsx>{`
        @keyframes loading {
          0%, 80%, 100% { transform: scale(0); }
          40% { transform: scale(1.0); }
        }
      `}</style>
    </div>
  );

  return (
    <div>
      {step === 1 && renderAuthForm()}
      {step === 2 && renderBackgroundForm()}
    </div>
  );
};

export default AuthComponent;