import React from 'react';
import Layout from '@theme/Layout';
import AuthComponent from '../components/AuthComponent';

export default function AuthPage() {
  return (
    <Layout title="Authentication" description="Sign up or sign in to personalize your learning experience">
      <div style={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        padding: '20px'
      }}>
        <div style={{
          background: 'rgba(255, 255, 255, 0.1)',
          backdropFilter: 'blur(10px)',
          borderRadius: '20px',
          padding: '40px',
          maxWidth: '500px',
          width: '100%',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.1)',
          border: '1px solid rgba(255, 255, 255, 0.18)'
        }}>
          <div style={{ textAlign: 'center', marginBottom: '30px' }}>
            <h1 style={{
              color: 'white',
              fontSize: '28px',
              fontWeight: '700',
              margin: '0 0 10px 0'
            }}>
              🤖 Learn Humanoid Robotics by
              
               Agentic Sphere.
            </h1>
            <p style={{
              color: 'rgba(255, 255, 255, 0.8)',
              fontSize: '16px',
              margin: '0'
            }}>
              Join our learning platform to access personalized content and features.
            </p>
          </div>
          <AuthComponent />
          <div style={{
            textAlign: 'center',
            marginTop: '20px',
            color: 'rgba(255, 255, 255, 0.6)',
            fontSize: '14px'
          }}>
            <p>🔒 Your data is secure and private</p>
          </div>
        </div>
      </div>
    </Layout>
  );
}