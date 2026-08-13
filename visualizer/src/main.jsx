import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ConfigProvider, theme } from 'antd'
import './index.css'
import App from './App.jsx'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ConfigProvider
      theme={{
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: '#4a90e2',
          colorBgBase: '#0a0e17',
          colorBgContainer: '#111827',
          colorBgElevated: '#1a2233',
          colorBorder: '#2d3a4f',
          colorText: '#f0f4f8',
          colorTextSecondary: '#94a3b8',
          fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
          borderRadius: 10,
          colorSuccess: '#34d399',
          colorWarning: '#fbbf24',
          colorError: '#f87171',
          colorLink: '#4a90e2',
        },
        components: {
          Layout: {
            headerBg: 'rgba(10, 14, 23, 0.92)',
            bodyBg: '#0a0e17',
            siderBg: '#111827',
          },
          Menu: {
            darkItemBg: 'transparent',
            darkSubMenuItemBg: '#111827',
            darkItemSelectedBg: 'rgba(74, 144, 226, 0.15)',
            darkItemHoverBg: '#1a2233',
          },
          Card: {
            colorBgContainer: '#1a2233',
            colorBorderSecondary: '#2d3a4f',
          },
        },
      }}
    >
      <App />
    </ConfigProvider>
  </StrictMode>,
)
