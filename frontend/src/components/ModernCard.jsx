import React from 'react';

const ModernCard = ({ 
  children, 
  className = '', 
  padding = true,
  shadow = true,
  ...props 
}) => {
  const cardClasses = `
    bg-white rounded-xl border border-gray-200
    ${shadow ? 'shadow-sm hover:shadow-md' : ''}
    ${padding ? 'p-6' : ''}
    transition-shadow duration-200
    ${className}
  `;

  return (
    <div className={cardClasses} {...props}>
      {children}
    </div>
  );
};

export default ModernCard;