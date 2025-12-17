import React from 'react';
import NavbarItem from '@theme/NavbarItem';
import CompleteWebsiteTranslation from '../../components/CompleteWebsiteTranslation';

// Fixed translation navbar item with complete website translation functionality
const NavbarItemCustomTranslationNavbarButton = (props) => {
  return (
    <div className="navbar__item navbar__item--right">
      <CompleteWebsiteTranslation />
    </div>
  );
};

export default NavbarItemCustomTranslationNavbarButton;