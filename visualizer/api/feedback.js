import { Resend } from "resend";

const resend = new Resend(process.env.RESEND_API_KEY);

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { name, email, feedback } = req.body;

  // Basic validation
  if (!name || !email || !feedback) {
    return res.status(400).json({ error: "All fields are required" });
  }

  try {
    await resend.emails.send({
      from: "InferX Feedback <onboarding@resend.dev>",
      to: "kumarmanikanta808@gmail.com",
      subject: `InferX Feedback from ${name}`,
      replyTo: email,
      html: `
        <div style="font-family: -apple-system, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #4a90e2;">New Feedback — InferX Visualizer</h2>
          <table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
            <tr>
              <td style="padding: 8px 12px; font-weight: 600; color: #555; width: 80px;">Name</td>
              <td style="padding: 8px 12px;">${name}</td>
            </tr>
            <tr>
              <td style="padding: 8px 12px; font-weight: 600; color: #555;">Email</td>
              <td style="padding: 8px 12px;"><a href="mailto:${email}">${email}</a></td>
            </tr>
          </table>
          <div style="background: #f5f7fa; padding: 16px; border-radius: 8px; margin-top: 12px;">
            <p style="font-weight: 600; color: #333; margin-bottom: 8px;">Feedback:</p>
            <p style="color: #555; white-space: pre-wrap; line-height: 1.6;">${feedback}</p>
          </div>
          <p style="margin-top: 20px; font-size: 12px; color: #999;">Sent from InferX Visualizer feedback widget</p>
        </div>
      `,
    });

    return res.status(200).json({ success: true });
  } catch (error) {
    console.error("Feedback email error:", error);
    return res.status(500).json({ error: "Failed to send feedback" });
  }
}
