import React from 'react';

const StepIndicator = ({ currentStep, steps }) => {
  return (
    <div className="flex items-center justify-center mb-8">
      {steps.map((step, index) => {
        const stepNumber = index + 1;
        const isActive = stepNumber === currentStep;
        const isCompleted = stepNumber < currentStep;
        
        return (
          <div key={stepNumber} className="flex items-center">
            {/* Step Circle */}
            <div className="flex items-center">
              <div
                className={`
                  flex items-center justify-center w-10 h-10 rounded-full border-2 font-semibold text-sm
                  transition-all duration-200
                  ${isCompleted 
                    ? 'bg-emerald-600 border-emerald-600 text-white' 
                    : isActive 
                      ? 'bg-blue-600 border-blue-600 text-white' 
                      : 'bg-white border-gray-300 text-gray-500'
                  }
                `}
              >
                {isCompleted ? (
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                ) : (
                  stepNumber
                )}
              </div>
              
              {/* Step Label */}
              <div className="ml-3">
                <div
                  className={`
                    text-sm font-medium
                    ${isActive ? 'text-blue-600' : isCompleted ? 'text-emerald-600' : 'text-gray-500'}
                  `}
                >
                  {step.title}
                </div>
                {step.description && (
                  <div className="text-xs text-gray-400 mt-1">
                    {step.description}
                  </div>
                )}
              </div>
            </div>
            
            {/* Connector Line */}
            {index < steps.length - 1 && (
              <div
                className={`
                  w-12 h-0.5 mx-4
                  ${stepNumber < currentStep ? 'bg-emerald-600' : 'bg-gray-300'}
                  transition-colors duration-200
                `}
              />
            )}
          </div>
        );
      })}
    </div>
  );
};

export default StepIndicator;