import React, { useEffect, useState } from 'react';
import Navbar from '@theme-original/Navbar';
import { useUser } from '../contexts/UserContext';
import Link from '@docusaurus/Link';

const NavbarWrapper = (props) => {
  const { user, logout } = useUser();
  const [items, setItems] = useState(props.items);

  useEffect(() => {
    // Clone the original navbar items
    const originalItems = props.items || [];
    let updatedItems = [...originalItems];

    // Find the Book link and restrict access if user is not authenticated
    const bookItemIndex = updatedItems.findIndex(item => item.label === 'Book');
    if (bookItemIndex !== -1) {
      if (!user) {
        // If user is not authenticated, redirect to auth page when clicking Book
        updatedItems[bookItemIndex] = {
          ...updatedItems[bookItemIndex],
          to: '/auth',
          // Change label to indicate user needs to sign up
          label: 'Sign Up to Access Book'
        };
      }
    }

    // Find and update the auth link based on user status
    const authItemIndex = updatedItems.findIndex(item => item.label === 'Sign In');

    if (authItemIndex !== -1) {
      if (user) {
        // Replace with user profile/logout options
        updatedItems[authItemIndex] = {
          type: 'dropdown',
          label: user.name || user.email || 'Account',
          position: 'right',
          items: [
            {
              label: 'Book Directory',
              to: '/docs/intro', // This will take them to the book directory
            },
            {
              label: 'Sign Out',
              to: '#',
              onClick: (e) => {
                e.preventDefault();
                logout();
              }
            }
          ]
        };
      } else {
        // Ensure it's the sign in link
        updatedItems[authItemIndex] = {
          ...updatedItems[authItemIndex],
          label: 'Sign In',
          to: '/auth'
        };
      }
    } else {
      // If we don't find the specific auth link, we might need to add it
      if (user) {
        // Add user dropdown if not found and user is logged in
        const hasUserDropdown = updatedItems.some(item => item.label === (user.name || user.email || 'Account'));
        if (!hasUserDropdown) {
          updatedItems.push({
            type: 'dropdown',
            label: user.name || user.email || 'Account',
            position: 'right',
            items: [
              {
                label: 'Book Directory',
                to: '/docs/intro',
              },
              {
                label: 'Sign Out',
                to: '#',
                onClick: (e) => {
                  e.preventDefault();
                  logout();
                }
              }
            ]
          });
        }
      }
    }


    setItems(updatedItems);
  }, [user, logout, props.items]);

  return (
    <Navbar {...props} items={items} />
  );
};

export { NavbarWrapper };

// Render the Navbar and a floating translate button so it's available site-wide
export default NavbarWrapper;