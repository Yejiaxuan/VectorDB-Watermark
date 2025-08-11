import React, { forwardRef } from 'react';

const ModernInput = forwardRef(({ 
  label, 
  error, 
  className = '', 
  required = false,
  ...props 
}, ref) => {
  const inputClasses = `
    w-full px-4 py-3 border rounded-lg transition-all duration-200 
    focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
    ${error 
      ? 'border-red-300 bg-red-50 focus:ring-red-500' 
      : 'border-gray-300 bg-white hover:border-gray-400'
    }
    ${className}
  `;

  return (
    <div className="space-y-2">
      {label && (
        <label className="block text-sm font-semibold text-gray-700">
          {label}
          {required && <span className="text-red-500 ml-1">*</span>}
        </label>
      )}
      <input
        ref={ref}
        className={inputClasses}
        {...props}
      />
      {error && (
        <p className="text-sm text-red-600 flex items-center">
          <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16c-.77.833.192 2.5 1.732 2.5z" />
          </svg>
          {error}
        </p>
      )}
    </div>
  );
});

ModernInput.displayName = 'ModernInput';

export default ModernInput;