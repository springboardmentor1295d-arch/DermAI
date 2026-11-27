// DermAI mobile: camera / gallery -> quality check -> confirm -> upload -> analysis.
// Set API to your computer's LAN IP, e.g. http://192.168.1.20:8000 (phone and PC on the same Wi-Fi).
import React, { useRef, useState } from "react";
import { ActivityIndicator, Alert, Button, Image, ScrollView, Text, TextInput, View } from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as ImagePicker from "expo-image-picker";

const API = "http://192.168.1.20:8000";

export default function App() {
  const [token, setToken] = useState(null);
  const [email, setEmail] = useState(""); const [pw, setPw] = useState("");
  const [perm, requestPerm] = useCameraPermissions();
  const cam = useRef(null);
  const [showCam, setShowCam] = useState(false);
  const [photo, setPhoto] = useState(null); const [quality, setQuality] = useState(null);
  const [result, setResult] = useState(null); const [busy, setBusy] = useState(false);
  const auth = { Authorization: `Bearer ${token}` };

  const login = async () => {
    const r = await fetch(`${API}/auth/login`, { method: "POST", body: new URLSearchParams({ username: email, password: pw }) });
    if (!r.ok) return Alert.alert("Login failed");
    setToken((await r.json()).access_token);
  };
  const form = (uri, extra = {}) => {
    const f = new FormData();
    f.append("file", { uri, name: "lesion.jpg", type: "image/jpeg" });
    Object.entries(extra).forEach(([k, v]) => f.append(k, String(v)));
    return f;
  };
  const checkQuality = async (uri, source) => {
    setPhoto({ uri, source }); setResult(null); setQuality(null);
    const r = await fetch(`${API}/quality/check`, { method: "POST", headers: auth, body: form(uri) });
    setQuality(await r.json());
  };
  const take = async () => { const p = await cam.current.takePictureAsync({ quality: 0.9 }); setShowCam(false); checkQuality(p.uri, "camera"); };
  const pick = async () => {
    const r = await ImagePicker.launchImageLibraryAsync({ mediaTypes: ImagePicker.MediaTypeOptions.Images, quality: 0.9, allowsEditing: true });
    if (!r.canceled) checkQuality(r.assets[0].uri, "gallery");
  };
  const analyze = async (override = false) => {
    setBusy(true);
    try {
      const c = await (await fetch(`${API}/cases`, { method: "POST", headers: { ...auth, "Content-Type": "application/json" }, body: JSON.stringify({ patient_code: "MOBILE" }) })).json();
      const up = await fetch(`${API}/cases/${c.id}/images`, { method: "POST", headers: auth, body: form(photo.uri, { source: photo.source, allow_poor: override }) });
      if (!up.ok) throw new Error((await up.json()).detail?.message || "Upload failed");
      const img = await up.json();
      const an = await fetch(`${API}/images/${img.id}/analyze?force=${override}`, { method: "POST", headers: auth });
      if (!an.ok) throw new Error((await an.json()).detail);
      setResult(await an.json());
    } catch (e) { Alert.alert("Error", String(e.message || e)); }
    setBusy(false);
  };

  if (!token) return (
    <View style={{ padding: 24, marginTop: 80 }}><Text style={{ fontSize: 26, fontWeight: "700" }}>DermAI</Text>
      <TextInput placeholder="Email" autoCapitalize="none" value={email} onChangeText={setEmail} style={{ borderWidth: 1, padding: 8, marginVertical: 8 }} />
      <TextInput placeholder="Password" secureTextEntry value={pw} onChangeText={setPw} style={{ borderWidth: 1, padding: 8, marginBottom: 8 }} />
      <Button title="Log in" onPress={login} /></View>);

  if (showCam) return (
    <CameraView ref={cam} style={{ flex: 1 }} facing="back">
      <View style={{ flex: 1, justifyContent: "flex-end", padding: 24 }}>
        <Text style={{ color: "#fff", marginBottom: 8 }}>Good light, lesion centred, 10–15 cm away, hold steady.</Text>
        <Button title="Capture" onPress={take} /></View></CameraView>);

  return (
    <ScrollView contentContainerStyle={{ padding: 20, paddingTop: 60 }}>
      <Text style={{ color: "#b45309" }}>AI-assisted analysis. Not a medical diagnosis.</Text>
      <View style={{ flexDirection: "row", gap: 8, marginVertical: 12 }}>
        <Button title="Camera" onPress={async () => { if (!perm?.granted) await requestPerm(); setShowCam(true); }} />
        <Button title="Gallery" onPress={pick} /></View>
      {photo && <Image source={{ uri: photo.uri }} style={{ width: "100%", height: 260, borderRadius: 8 }} />}
      {quality && <View style={{ marginVertical: 8 }}>
        <Text style={{ fontWeight: "700" }}>Quality: {quality.label}</Text>
        {quality.issues.map((i) => <Text key={i} style={{ color: "#b91c1c" }}>• {i}</Text>)}
        <Button title={busy ? "Analyzing…" : quality.label === "Poor" ? "Analyze anyway (flagged)" : "Confirm & analyze"} disabled={busy} onPress={() => analyze(quality.label === "Poor")} /></View>}
      {busy && <ActivityIndicator />}
      {result && <View style={{ marginTop: 12 }}>
        <Text style={{ fontSize: 20, fontWeight: "700" }}>{result.predicted_name}</Text>
        <Text>Confidence {(result.confidence * 100).toFixed(1)}% · {result.risk}</Text>
        {result.needs_review && <Text style={{ color: "#b91c1c" }}>Clinical review recommended.</Text>}
        <Image source={{ uri: API + result.cam_url }} style={{ width: "100%", height: 240, marginTop: 8 }} />
        <Text>{result.top3.map(([c, p]) => `${c} ${(p * 100).toFixed(0)}%`).join("  ·  ")}</Text></View>}
    </ScrollView>);
}
