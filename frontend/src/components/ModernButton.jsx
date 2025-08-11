import React from 'react';

const ModernButton = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  loading = false, 
  disabled = false,
  className = '',
  ...props 
}) => {
  const getVariantStyles = () => {
    switch (variant) {
      case 'primary':
        return 'bg-blue-600 hover:bg-blue-700 text-white border-transparent';
      case 'secondary':
        return 'bg-gray-600 hover:bg-gray-700 text-white border-transparent';
      case 'outline':
        return 'bg-transparent hover:bg-gray-50 text-gray-700 border-gray-300 hover:border-gray-400';
      case 'success':
        return 'bg-emerald-600 hover:bg-emerald-700 text-white border-transparent';
      case 'warning':
        return 'bg-amber-600 hover:bg-amber-700 text-white border-transparent';
      case 'danger':
        return 'bg-red-600 hover:bg-red-700 text-white border-transparent';
      default:
        return 'bg-blue-600 hover:bg-blue-700 text-white border-transparent';
    }
  };

  const getSizeStyles = () => {
    switch (size) {
      case 'sm':
        return 'px-3 py-2 text-sm';
      case 'md':
        return 'px-4 py-2.5 text-sm';
      case 'lg':
        return 'px-6 py-3 text-base';
      default:
        return 'px-4 py-2.5 text-sm';
    }
  };

  const baseStyles = `
    inline-flex items-center justify-center font-medium rounded-lg border
    transition-all duration-200 ease-in-out
    focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500
    disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-current
  `;

  const buttonClasses = `
    ${baseStyles}
    ${getVariantStyles()}
    ${getSizeStyles()}
    ${className}
  `;

  return (
    <button
      className={buttonClasses}
      disabled={disabled || loading}
      {...props}
    >
      {loading && (
        <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
        </svg>
      )}
      {children}
    </button>
  );
};

export default ModernButton;