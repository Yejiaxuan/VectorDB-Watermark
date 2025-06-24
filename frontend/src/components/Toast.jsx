import React, { useEffect, useState } from 'react';

const Toast = ({ message, type = 'success', isVisible, onClose, position = 'bottom' }) => {
  const [show, setShow] = useState(false);

  useEffect(() => {
    if (isVisible) {
      setShow(true);
      const timer = setTimeout(() => {
        setShow(false);
        setTimeout(onClose, 300); // 等待动画完成后调用onClose
      }, 3000);
      
      return () => clearTimeout(timer);
    }
  }, [isVisible, onClose]);

  if (!isVisible && !show) return null;

  const positionClasses = {
    top: "fixed top-6 right-6",
    bottom: "fixed bottom-6 left-1/2 transform -translate-x-1/2"
  };

  const baseClasses = `${positionClasses[position]} max-w-sm p-4 rounded-xl shadow-lg backdrop-blur-lg transform transition-all duration-300 ease-in-out z-50`;
  
  const typeClasses = {
    success: "bg-white/90 border border-green-200",
    error: "bg-white/90 border border-red-200",
    warning: "bg-white/90 border border-yellow-200",
    info: "bg-white/90 border border-blue-200"
  };
  
  const iconClasses = {
    success: "text-green-500",
    error: "text-red-500", 
    warning: "text-yellow-500",
    info: "text-blue-500"
  };

  const icons = {
    success: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    ),
    error: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
      </svg>
    ),
    warning: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
      </svg>
    ),
    info: (
      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
      </svg>
    )
  };

  return (
    <div 
      className={`${baseClasses} ${typeClasses[type]} ${
        show 
          ? position === 'bottom' 
            ? 'translate-y-0 opacity-100' 
            : 'translate-x-0 opacity-100'
          : position === 'bottom'
            ? 'translate-y-full opacity-0'
            : 'translate-x-full opacity-0'
      }`}
    >
      <div className="flex items-center">
        <div className={`flex-shrink-0 ${iconClasses[type]} mr-3`}>
          {icons[type]}
        </div>
        <div className="flex-1">
          <p className="text-gray-800 text-sm font-medium leading-5">
            {message}
          </p>
        </div>
        <button
          onClick={() => {
            setShow(false);
            setTimeout(onClose, 300);
          }}
          className="flex-shrink-0 ml-2 text-gray-400 hover:text-gray-600 transition-colors duration-150 ease-in-out"
        >
          <span className="sr-only">关闭</span>
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
};

export default Toast; 