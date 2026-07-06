import type { Metadata } from "next";
import { Inter, Outfit } from "next/font/google";
import "./globals.css";
import TopNav from "@/components/TopNav";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const outfit = Outfit({
  variable: "--font-outfit",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "VigilantVision Dashboard",
  description: "Advanced VigilantVision AI Security System Dashboard",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${outfit.variable} font-sans antialiased flex flex-col bg-background text-foreground min-h-screen`}>
        <TopNav />
        <main className="flex-1 px-4 pt-2 pb-4 flex flex-col">
          {children}
        </main>
      </body>
    </html>
  );
}
