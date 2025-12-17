import React from 'react';
import { useUser } from '../contexts/UserContext';
import Link from '@docusaurus/Link';

const Navbar = () => {
  const { user, logout } = useUser();

  return (
    <div className="navbar-user-section">
      {user ? (
        <div className="user-menu">
          <span className="user-name">Hello, {user.name || user.email}</span>
          <button
            className="logout-button"
            onClick={logout}
            style={{
              marginLeft: '15px',
              padding: '5px 10px',
              backgroundColor: '#007cba',
              color: 'white',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            Sign Out
          </button>
        </div>
      ) : (
        <Link
          to="/auth"
          className="signin-link"
          style={{
            padding: '5px 10px',
            backgroundColor: '#007cba',
            color: 'white',
            textDecoration: 'none',
            borderRadius: '4px'
          }}
        >
          Sign In
        </Link>
      )}
    </div>
  );
};

export default Navbar;