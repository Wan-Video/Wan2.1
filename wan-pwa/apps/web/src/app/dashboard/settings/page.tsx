import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function SettingsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Settings</h1>
        <p className="text-muted-foreground">Manage your account and preferences</p>
      </div>

      <div className="grid gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Profile</CardTitle>
            <CardDescription>Manage your profile information</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">Profile settings coming soon...</p>
          </CardContent>
        </Card>

        <Card>
          <CardTitle className="p-6 pb-4">Billing & Credits</CardTitle>
          <CardDescription className="px-6 pb-6">
            Manage your credits and subscription
          </CardDescription>
          <CardContent>
            <p className="text-sm text-muted-foreground">Billing settings coming soon...</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>API Keys</CardTitle>
            <CardDescription>Manage your API access</CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-muted-foreground">API key management coming soon...</p>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
