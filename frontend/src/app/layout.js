import { Geist, Geist_Mono } from 'next/font/google';
import './globals.css';
import NavBar from '@/components/nav/NavBar';
import Providers from '@/components/Providers';
import TrainingDrawer from '@/components/training/TrainingDrawer';
import Toast from '@/components/training/Toast';

const geistSans = Geist({ variable: '--font-geist-sans', subsets: ['latin'] });
const geistMono = Geist_Mono({ variable: '--font-geist-mono', subsets: ['latin'] });

export const metadata = {
  title: 'Jamuna Bankline Prediction',
  description: 'TFT-powered bankline shift forecasting for 50 reaches of the Jamuna River',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="dark">
      <body className={`${geistSans.variable} ${geistMono.variable} antialiased min-h-screen`}
            style={{ background: 'var(--bg)', color: 'var(--text)' }}>
        <Providers>
          <NavBar />
          <main className="pt-16">
            {children}
          </main>
          <TrainingDrawer />
          <Toast />
        </Providers>
      </body>
    </html>
  );
}
