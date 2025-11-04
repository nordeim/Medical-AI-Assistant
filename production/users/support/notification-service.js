// Healthcare Notification Service
// Multi-channel notification system for user management events

const nodemailer = require('nodemailer');
const twilio = require('twilio');
const crypto = require('crypto');

class HealthcareNotificationService {
  constructor() {
    this.emailTransporter = this.initializeEmailTransporter();
    this.smsClient = this.initializeSMSClient();
    this.pushNotifications = this.initializePushNotifications();
    this.notificationQueue = [];
    this.deliveryTracking = new Map();
    this.templateEngine = require('./notification-templates');
  }

  // Send Email Notification
  async sendEmail(emailData) {
    try {
      const {
        recipient,
        subject,
        content,
        template,
        templateData,
        priority = 'normal',
        medicalContext = false,
        attachment
      } = emailData;

      // Select appropriate template
      const emailContent = template ? 
        await this.templateEngine.renderTemplate(template, templateData) :
        content;

      // Create email message
      const mailOptions = {
        from: this.getSenderEmail(medicalContext),
        to: recipient.email,
        subject: this.addMedicalContextPrefix(subject, medicalContext),
        html: emailContent.html || emailContent,
        text: emailContent.text || this.stripHtml(emailContent.html || emailContent),
        priority,
        headers: {
          'X-Healthcare-System': 'true',
          'X-Medical-Context': medicalContext ? 'true' : 'false',
          'X-Notification-ID': crypto.randomUUID()
        }
      };

      // Add attachment if provided
      if (attachment) {
        mailOptions.attachments = [{
          filename: attachment.filename,
          content: attachment.content,
          contentType: attachment.contentType
        }];
      }

      // Send email
      const result = await this.emailTransporter.sendMail(mailOptions);

      // Log notification
      await this.logNotification('email', {
        recipient,
        subject,
        template,
        messageId: result.messageId,
        medicalContext,
        priority
      });

      return {
        success: true,
        messageId: result.messageId,
        channel: 'email',
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      console.error('Email notification error:', error);
      throw error;
    }
  }

  // Send SMS Notification
  async sendSMS(smsData) {
    try {
      const {
        recipient,
        message,
        template,
        templateData,
        priority = 'normal',
        medicalContext = false,
        language = 'en'
      } = smsData;

      // Render template if provided
      const smsContent = template ? 
        await this.templateEngine.renderTemplate(template, templateData) :
        message;

      // Add medical context prefix for urgent notifications
      const prefixedMessage = this.addMedicalContextPrefix(smsContent, medicalContext);

      // Send SMS
      const result = await this.smsClient.messages.create({
        body: prefixedMessage,
        from: process.env.TWILIO_PHONE_NUMBER,
        to: recipient.phone,
        statusCallback: process.env.SMS_WEBHOOK_URL
      });

      // Log notification
      await this.logNotification('sms', {
        recipient,
        message: smsContent,
        template,
        messageSid: result.sid,
        medicalContext,
        priority
      });

      return {
        success: true,
        messageSid: result.sid,
        channel: 'sms',
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      console.error('SMS notification error:', error);
      throw error;
    }
  }

  // Send Push Notification
  async sendPushNotification(pushData) {
    try {
      const {
        recipient,
        title,
        message,
        template,
        templateData,
        priority = 'normal',
        medicalContext = false,
        data,
        actionButtons
      } = pushData;

      // Render content from template if provided
      const notificationContent = template ? 
        await this.templateEngine.renderTemplate(template, templateData) :
        { title, message };

      // Create push notification payload
      const pushPayload = {
        title: this.addMedicalContextPrefix(notificationContent.title, medicalContext),
        body: notificationContent.message,
        data: {
          ...data,
          medicalContext,
          notificationType: 'healthcare_system',
          timestamp: new Date().toISOString()
        },
        priority,
        actions: actionButtons || [],
        badge: await this.getUnreadCount(recipient.userId)
      };

      // Send to all user's devices
      const devices = await this.getUserDevices(recipient.userId);
      const results = [];

      for (const device of devices) {
        try {
          const result = await this.sendToDevice(device, pushPayload);
          results.push(result);
        } catch (error) {
          console.error(`Push notification failed for device ${device.deviceId}:`, error);
          results.push({ success: false, deviceId: device.deviceId, error: error.message });
        }
      }

      // Log notification
      await this.logNotification('push', {
        recipient,
        title: notificationContent.title,
        template,
        deviceCount: devices.length,
        successfulDeliveries: results.filter(r => r.success).length,
        medicalContext,
        priority
      });

      return {
        success: true,
        results,
        channel: 'push',
        timestamp: new Date().toISOString()
      };

    } catch (error) {
      console.error('Push notification error:', error);
      throw error;
    }
  }

  // Send Notification to Role
  async sendToRole(roleName, notificationData) {
    try {
      // Get all users with the specified role
      const users = await this.getUsersByRole(roleName);

      if (users.length === 0) {
        console.log(`No users found with role: ${roleName}`);
        return { success: true, recipientCount: 0 };
      }

      const results = [];
      
      for (const user of users) {
        try {
          const result = await this.sendNotification(user, notificationData);
          results.push(result);
        } catch (error) {
          console.error(`Failed to send notification to user ${user.userId}:`, error);
          results.push({ success: false, userId: user.userId, error: error.message });
        }
      }

      return {
        success: true,
        recipientCount: users.length,
        successfulDeliveries: results.filter(r => r.success).length,
        failedDeliveries: results.filter(r => !r.success).length,
        results
      };

    } catch (error) {
      console.error('Role-based notification error:', error);
      throw error;
    }
  }

  // Send Medical Emergency Notification
  async sendMedicalEmergencyNotification(emergencyData) {
    try {
      const {
        patientId,
        emergencyType,
        severity,
        location,
        requestingUser,
        medicalSpecialty,
        message
      } = emergencyData;

      // Critical emergency notification template
      const emergencyNotification = {
        type: 'medical_emergency',
        priority: 'critical',
        title: `MEDICAL EMERGENCY - ${emergencyType}`,
        message,
        data: {
          patientId,
          emergencyType,
          severity,
          location,
          requestingUser: requestingUser.userId,
          medicalSpecialty,
          timestamp: new Date().toISOString()
        }
      };

      // Send to emergency response team
      const emergencyTeam = await this.getEmergencyResponseTeam(patientId, medicalSpecialty);
      
      const results = [];
      
      for (const teamMember of emergencyTeam) {
        try {
          // Send to all channels for critical emergencies
          const emailResult = await this.sendEmail({
            recipient: teamMember,
            subject: emergencyNotification.title,
            template: 'medical_emergency',
            templateData: emergencyNotification.data,
            priority: 'critical',
            medicalContext: true
          });

          const smsResult = await this.sendSMS({
            recipient: teamMember,
            message: `URGENT: ${emergencyNotification.title} - ${location}`,
            template: 'medical_emergency_sms',
            templateData: emergencyNotification.data,
            priority: 'critical',
            medicalContext: true
          });

          results.push({ email: emailResult, sms: smsResult });
        } catch (error) {
          console.error(`Emergency notification failed for ${teamMember.userId}:`, error);
          results.push({ success: false, userId: teamMember.userId, error: error.message });
        }
      }

      // Log emergency notification
      await this.logNotification('emergency', {
        emergencyType,
        severity,
        patientId,
        teamSize: emergencyTeam.length,
        timestamp: new Date().toISOString()
      });

      return {
        success: true,
        emergencyType,
        teamMembers: emergencyTeam.length,
        results
      };

    } catch (error) {
      console.error('Medical emergency notification error:', error);
      throw error;
    }
  }

  // Batch Notification Processing
  async processNotificationQueue() {
    if (this.notificationQueue.length === 0) return;

    const batch = this.notificationQueue.splice(0, 10); // Process 10 at a time

    for (const notification of batch) {
      try {
        await this.sendNotification(notification.recipient, notification.data);
      } catch (error) {
        console.error(`Failed to process queued notification:`, error);
        
        // Move to dead letter queue for retry
        await this.moveToDeadLetterQueue(notification, error.message);
      }
    }
  }

  // Notification Templates
  async renderTemplate(templateName, data, format = 'html') {
    return await this.templateEngine.renderTemplate(templateName, data, format);
  }

  // Notification Preferences Management
  async updateUserPreferences(userId, preferences) {
    try {
      const { channels, medicalAlerts, reminderSettings, quietHours } = preferences;

      const updatedPreferences = {
        userId,
        channels: channels || {},
        medicalAlerts: {
          enabled: medicalAlerts?.enabled ?? true,
          severity: medicalAlerts?.severity || ['high', 'critical'],
          specialties: medicalAlerts?.specialties || [],
          ...medicalAlerts
        },
        reminderSettings: {
          enabled: reminderSettings?.enabled ?? true,
          frequency: reminderSettings?.frequency || 'daily',
          ...reminderSettings
        },
        quietHours: quietHours || null,
        lastUpdated: new Date().toISOString()
      };

      await this.storeNotificationPreferences(updatedPreferences);

      return updatedPreferences;

    } catch (error) {
      console.error('Notification preferences update error:', error);
      throw error;
    }
  }

  // Helper Methods
  initializeEmailTransporter() {
    return nodemailer.createTransporter({
      host: process.env.SMTP_HOST,
      port: process.env.SMTP_PORT || 587,
      secure: process.env.SMTP_SECURE === 'true',
      auth: {
        user: process.env.SMTP_USER,
        pass: process.env.SMTP_PASS
      },
      tls: {
        minVersion: 'TLSv1.2'
      }
    });
  }

  initializeSMSClient() {
    if (!process.env.TWILIO_ACCOUNT_SID) {
      console.warn('Twilio not configured - SMS notifications will be disabled');
      return null;
    }

    return twilio(
      process.env.TWILIO_ACCOUNT_SID,
      process.env.TWILIO_AUTH_TOKEN
    );
  }

  initializePushNotifications() {
    // Initialize push notification service (Firebase, OneSignal, etc.)
    return {
      enabled: !!process.env.PUSH_SERVICE_CONFIG
    };
  }

  getSenderEmail(medicalContext) {
    const sender = medicalContext ? 
      process.env.MEDICAL_NOTIFICATIONS_EMAIL :
      process.env.DEFAULT_NOTIFICATIONS_EMAIL;
    
    return sender || 'noreply@healthcaresystem.com';
  }

  addMedicalContextPrefix(content, medicalContext) {
    if (!medicalContext) return content;
    
    const prefix = '[MEDICAL] ';
    return prefix + (typeof content === 'string' ? content : content.toString());
  }

  stripHtml(html) {
    return html.replace(/<[^>]*>/g, '').trim();
  }

  async sendNotification(recipient, notificationData) {
    const { channels } = await this.getNotificationPreferences(recipient.userId);
    
    const results = [];

    for (const [channel, enabled] of Object.entries(channels)) {
      if (!enabled || !notificationData[channel]) continue;

      try {
        let result;
        
        switch (channel) {
          case 'email':
            result = await this.sendEmail({
              recipient,
              ...notificationData.email
            });
            break;
          
          case 'sms':
            result = await this.sendSMS({
              recipient,
              ...notificationData.sms
            });
            break;
          
          case 'push':
            result = await this.sendPushNotification({
              recipient,
              ...notificationData.push
            });
            break;
          
          default:
            console.warn(`Unknown notification channel: ${channel}`);
            continue;
        }

        results.push({ channel, ...result });
      } catch (error) {
        console.error(`Failed to send ${channel} notification:`, error);
        results.push({ channel, success: false, error: error.message });
      }
    }

    return {
      recipient: recipient.userId,
      results,
      successfulChannels: results.filter(r => r.success).length,
      totalChannels: results.length
    };
  }

  // Database Operations
  async storeNotificationPreferences(preferences) {
    await require('../database/user-database').storeNotificationPreferences(preferences);
  }

  async getNotificationPreferences(userId) {
    return await require('../database/user-database').getNotificationPreferences(userId);
  }

  async getUsersByRole(roleName) {
    return await require('../database/user-database').getUsersByRole(roleName);
  }

  async getUserDevices(userId) {
    return await require('../database/user-database').getUserDevices(userId);
  }

  async getUnreadCount(userId) {
    // Get unread notification count
    return await require('../database/user-database').getUnreadNotificationCount(userId);
  }

  async getEmergencyResponseTeam(patientId, medicalSpecialty) {
    // Get appropriate emergency response team based on specialty and location
    return await require('../database/user-database').getEmergencyResponseTeam(patientId, medicalSpecialty);
  }

  async logNotification(channel, data) {
    await require('../database/user-database').logNotification(channel, data);
  }

  async sendToDevice(device, payload) {
    // Send push notification to specific device
    console.log(`Sending push notification to device ${device.deviceId}`);
    return { success: true, deviceId: device.deviceId };
  }

  async moveToDeadLetterQueue(notification, error) {
    // Move failed notification to dead letter queue for retry
    await require('../database/user-database').moveToDeadLetterQueue(notification, error);
  }
}

// Start queue processor
setInterval(() => {
  if (global.notificationService) {
    global.notificationService.processNotificationQueue();
  }
}, 30000); // Process every 30 seconds

module.exports = new HealthcareNotificationService();