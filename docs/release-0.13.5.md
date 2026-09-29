# OpenWeights 0.13.5

Dashboard sign-up and password reset work with email confirmation.

- Sign-up no longer signs the user in straight away when the project requires
  email confirmation; it tells them to open the confirmation link, which returns
  them to `/organizations`.
- A password-recovery link always ends on `/reset-password`, even if Supabase
  redirected it to the Site URL. Before, the user simply looked logged in, so
  the reset email behaved like a login link.

The hosted project's auth settings changed with this release (not in the repo):
Site URL `https://openweights.nielsrolf.com`, redirect allowlist
`https://openweights.nielsrolf.com/**`, custom SMTP via Resend from
`noreply@mail.nielsrolf.com`, and "Confirm email" turned on.

SDK and worker images are unchanged; `IMAGE_VERSION` stays v0.13.4.
