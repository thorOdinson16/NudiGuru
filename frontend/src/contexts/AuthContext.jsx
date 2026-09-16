import { createContext, useContext, useEffect, useMemo, useState } from "react";

import * as client from "@/api/client";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let active = true;

    if (!client.getToken()) {
      setLoading(false);
      return () => {
        active = false;
      };
    }

    client
      .fetchMe()
      .then((me) => {
        if (active) setUser(me);
      })
      .catch(() => {
        client.setToken(null);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, []);

  const value = useMemo(
    () => ({
      user,
      loading,
      async login(email, password) {
        const data = await client.login(email, password);
        client.setToken(data.access_token);
        setUser(data.user);
        return data.user;
      },
      async register(email, password, fullName) {
        const data = await client.register(email, password, fullName);
        client.setToken(data.access_token);
        setUser(data.user);
        return data.user;
      },
      logout() {
        client.setToken(null);
        setUser(null);
      },
    }),
    [user, loading]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
