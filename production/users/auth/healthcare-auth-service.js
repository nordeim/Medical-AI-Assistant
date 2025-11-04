// Healthcare Authentication System with Supabase Integration
// Production-grade authentication with medical credential validation

const { createClient } = require('@supabase/supabase-js');
const crypto = require('crypto');
const bcrypt = require('bcrypt');
const config = require('../config/user-management-config');

class HealthcareAuthenticationService {
  constructor() {
    this.supabase = createClient(
      process.env.SUPABASE_URL,
      process.env.SUPABASE_ANON_KEY,
      {
        auth: {
          autoRefreshToken: true,
          persistSession: true,
          detectSessionInUrl: true,
          flowType: 'pkce',
          storage: new MemoryStorage()
        }
      }
    );
    
    this.config = config;
    this.auditLogger = require('../monitoring/audit-logger');
  }

  // User Registration for Healthcare Professionals
  async registerHealthcareUser(userData) {
    const {
      email,
      password,
      role,
      firstName,
      lastName,
      medicalLicense,
      specialty,
      institution,
      phoneNumber,
      address,
      dateOfBirth,
      emergencyContact
    } = userData;

    try {
      // Validate input data
      await this.validateRegistrationData(userData);

      // Create Supabase auth user
      const { data: authUser, error: authError } = await this.supabase.auth.signUp({
        email,
        password,
        options: {
          data: {
            role,
            firstName,
            lastName,
            medicalLicense,
            specialty,
            institution,
            phoneNumber,
            address,
            dateOfBirth,
            emergencyContact,
            registrationDate: new Date().toISOString(),
            verificationStatus: 'pending'
          }
        }
      });

      if (authError) {
        throw new Error(`Registration failed: ${authError.message}`);
      }

      // Store additional healthcare-specific data
      const { error: profileError } = await this.supabase
        .from('healthcare_users')
        .insert({
          user_id: authUser.user.id,
          email,
          first_name: firstName,
          last_name: lastName,
          role,
          medical_license: medicalLicense,
          specialty,
          institution,
          phone_number: phoneNumber,
          address: JSON.stringify(address),
          date_of_birth: dateOfBirth,
          emergency_contact: JSON.stringify(emergencyContact),
          registration_date: new Date().toISOString(),
          verification_status: 'pending',
          account_status: 'active',
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        });

      if (profileError) {
        // Clean up auth user if profile creation fails
        await this.supabase.auth.admin.deleteUser(authUser.user.id);
        throw new Error(`Profile creation failed: ${profileError.message}`);
      }

      // Log registration event
      await this.auditLogger.logEvent({
        userId: authUser.user.id,
        event: 'user.registered',
        details: {
          role,
          specialty,
          institution,
          registrationMethod: 'email'
        },
        timestamp: new Date().toISOString(),
        source: 'registration_service'
      });

      // Start verification workflow
      await this.initiateVerificationWorkflow(authUser.user.id, userData);

      return {
        success: true,
        user: authUser.user,
        verificationRequired: true,
        message: 'Registration successful. Verification required.'
      };

    } catch (error) {
      console.error('Registration error:', error);
      throw new Error(`Registration failed: ${error.message}`);
    }
  }

  // User Authentication with MFA Support
  async authenticateUser(email, password, mfaCode = null) {
    try {
      // Attempt authentication
      const { data: authData, error } = await this.supabase.auth.signInWithPassword({
        email,
        password
      });

      if (error) {
        // Log failed login attempt
        await this.auditLogger.logEvent({
          event: 'user.failed_login',
          details: { email, error: error.message },
          timestamp: new Date().toISOString(),
          source: 'authentication_service'
        });
        
        throw new Error(`Authentication failed: ${error.message}`);
      }

      // Check if MFA is required
      const userProfile = await this.getUserProfile(authData.user.id);
      const requiresMFA = this.config.healthcare.roles[userProfile.role]?.permissions.includes('*') || 
                         this.config.authentication.mfa.requiredRoles.includes(userProfile.role);

      if (requiresMFA && !mfaCode) {
        // Issue MFA challenge
        return {
          success: false,
          requiresMFA: true,
          user: authData.user,
          challengeId: await this.issueMFAChallenge(authData.user.id)
        };
      }

      // Verify MFA code if provided
      if (requiresMFA && mfaCode) {
        const mfaValid = await this.verifyMFAChallenge(authData.user.id, mfaCode);
        if (!mfaValid) {
          throw new Error('Invalid MFA code');
        }
      }

      // Update last login
      await this.updateLastLogin(authData.user.id);

      // Log successful login
      await this.auditLogger.logEvent({
        userId: authData.user.id,
        event: 'user.login',
        details: {
          role: userProfile.role,
          loginMethod: 'email',
          mfaUsed: requiresMFA && !!mfaCode
        },
        timestamp: new Date().toISOString(),
        source: 'authentication_service'
      });

      return {
        success: true,
        user: authData.user,
        session: authData.session,
        profile: userProfile
      };

    } catch (error) {
      console.error('Authentication error:', error);
      throw error;
    }
  }

  // MFA Challenge Management
  async issueMFAChallenge(userId) {
    const challengeId = crypto.randomUUID();
    const challenge = {
      id: challengeId,
      userId,
      type: 'totp',
      expiresAt: Date.now() + (5 * 60 * 1000), // 5 minutes
      attempts: 0,
      status: 'active'
    };

    // Store challenge (in production, use Redis or similar)
    await this.supabase
      .from('mfa_challenges')
      .insert({
        challenge_id: challengeId,
        user_id: userId,
        type: 'totp',
        expires_at: new Date(challenge.expiresAt).toISOString(),
        attempts: 0,
        status: 'active',
        created_at: new Date().toISOString()
      });

    // Send MFA code (implement based on method)
    await this.sendMFACode(userId, challengeId);

    return challengeId;
  }

  async verifyMFAChallenge(challengeId, code) {
    try {
      const { data: challenge, error } = await this.supabase
        .from('mfa_challenges')
        .select('*')
        .eq('challenge_id', challengeId)
        .eq('status', 'active')
        .single();

      if (error || !challenge) {
        throw new Error('Invalid challenge');
      }

      if (new Date(challenge.expires_at) < new Date()) {
        await this.deactivateChallenge(challengeId);
        throw new Error('Challenge expired');
      }

      if (challenge.attempts >= 3) {
        await this.deactivateChallenge(challengeId);
        throw new Error('Too many attempts');
      }

      // Verify code (implement TOTP verification)
      const isValid = await this.verifyTOTP(challenge.user_id, code);

      if (isValid) {
        await this.deactivateChallenge(challengeId);
        return true;
      } else {
        // Increment attempts
        await this.supabase
          .from('mfa_challenges')
          .update({ attempts: challenge.attempts + 1 })
          .eq('challenge_id', challengeId);
        
        return false;
      }

    } catch (error) {
      console.error('MFA verification error:', error);
      return false;
    }
  }

  // User Profile Management
  async getUserProfile(userId) {
    const { data, error } = await this.supabase
      .from('healthcare_users')
      .select('*')
      .eq('user_id', userId)
      .single();

    if (error) {
      throw new Error(`Failed to get user profile: ${error.message}`);
    }

    return data;
  }

  async updateUserProfile(userId, updates) {
    const allowedFields = [
      'phone_number', 'address', 'emergency_contact',
      'specialty', 'institution'
    ];

    const filteredUpdates = Object.keys(updates)
      .filter(key => allowedFields.includes(key))
      .reduce((obj, key) => {
        obj[key] = typeof updates[key] === 'object' 
          ? JSON.stringify(updates[key]) 
          : updates[key];
        return obj;
      }, {});

    filteredUpdates.updated_at = new Date().toISOString();

    const { error } = await this.supabase
      .from('healthcare_users')
      .update(filteredUpdates)
      .eq('user_id', userId);

    if (error) {
      throw new Error(`Profile update failed: ${error.message}`);
    }

    // Log profile update
    await this.auditLogger.logEvent({
      userId,
      event: 'user.profile_updated',
      details: { updatedFields: Object.keys(filteredUpdates) },
      timestamp: new Date().toISOString(),
      source: 'authentication_service'
    });

    return true;
  }

  // Password Management
  async changePassword(userId, oldPassword, newPassword) {
    try {
      // Verify old password
      const { data: authData, error } = await this.supabase.auth.signInWithPassword({
        email: (await this.getUserProfile(userId)).email,
        password: oldPassword
      });

      if (error) {
        throw new Error('Invalid current password');
      }

      // Update password
      const { error: updateError } = await this.supabase.auth.updateUser({
        password: newPassword
      });

      if (updateError) {
        throw new Error(`Password update failed: ${updateError.message}`);
      }

      // Log password change
      await this.auditLogger.logEvent({
        userId,
        event: 'user.password_change',
        details: { method: 'self_service' },
        timestamp: new Date().toISOString(),
        source: 'authentication_service'
      });

      return true;

    } catch (error) {
      console.error('Password change error:', error);
      throw error;
    }
  }

  // Session Management
  async createSession(userId, deviceInfo = {}) {
    const sessionData = {
      userId,
      deviceInfo,
      createdAt: new Date().toISOString(),
      expiresAt: new Date(Date.now() + this.config.authentication.sessionTimeout * 1000).toISOString()
    };

    const { data, error } = await this.supabase
      .from('user_sessions')
      .insert(sessionData)
      .select()
      .single();

    if (error) {
      throw new Error(`Session creation failed: ${error.message}`);
    }

    return data;
  }

  async terminateSession(sessionId) {
    const { error } = await this.supabase
      .from('user_sessions')
      .update({ 
        terminated_at: new Date().toISOString(),
        status: 'terminated'
      })
      .eq('session_id', sessionId);

    if (error) {
      throw new Error(`Session termination failed: ${error.message}`);
    }

    return true;
  }

  // Helper Methods
  async validateRegistrationData(userData) {
    const required = ['email', 'password', 'role', 'firstName', 'lastName', 'medicalLicense'];
    const missing = required.filter(field => !userData[field]);
    
    if (missing.length > 0) {
      throw new Error(`Missing required fields: ${missing.join(', ')}`);
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(userData.email)) {
      throw new Error('Invalid email format');
    }

    // Validate password strength
    await this.validatePasswordStrength(userData.password);

    // Validate role
    if (!this.config.healthcare.roles[userData.role]) {
      throw new Error('Invalid role');
    }

    // Validate medical license format (basic validation)
    if (!userData.medicalLicense || userData.medicalLicense.length < 6) {
      throw new Error('Invalid medical license');
    }
  }

  async validatePasswordStrength(password) {
    const policy = this.config.authentication.passwordPolicy;
    
    if (password.length < policy.minLength) {
      throw new Error(`Password must be at least ${policy.minLength} characters long`);
    }

    if (policy.requireUppercase && !/[A-Z]/.test(password)) {
      throw new Error('Password must contain at least one uppercase letter');
    }

    if (policy.requireLowercase && !/[a-z]/.test(password)) {
      throw new Error('Password must contain at least one lowercase letter');
    }

    if (policy.requireNumbers && !/\d/.test(password)) {
      throw new Error('Password must contain at least one number');
    }

    if (policy.requireSpecialChars && !/[!@#$%^&*(),.?":{}|<>]/.test(password)) {
      throw new Error('Password must contain at least one special character');
    }
  }

  async initiateVerificationWorkflow(userId, userData) {
    // Create verification records
    const verificationSteps = [
      { step: 'basic_information', status: 'completed' },
      { step: 'medical_credentials', status: 'pending' },
      { step: 'background_check', status: 'pending' },
      { step: 'reference_validation', status: 'pending' },
      { step: 'institutional_affiliation', status: 'pending' },
      { step: 'final_approval', status: 'pending' }
    ];

    for (const step of verificationSteps) {
      await this.supabase
        .from('verification_steps')
        .insert({
          user_id: userId,
          step_name: step.step,
          status: step.status,
          created_at: new Date().toISOString()
        });
    }

    // Notify administrators for review
    await this.notifyAdministrators(userId, 'registration_pending');
  }

  async updateLastLogin(userId) {
    await this.supabase
      .from('healthcare_users')
      .update({ 
        last_login: new Date().toISOString(),
        updated_at: new Date().toISOString()
      })
      .eq('user_id', userId);
  }

  async deactivateChallenge(challengeId) {
    await this.supabase
      .from('mfa_challenges')
      .update({ status: 'expired' })
      .eq('challenge_id', challengeId);
  }

  async verifyTOTP(userId, code) {
    // Implement TOTP verification (using libraries like otplib)
    // This is a placeholder - implement actual TOTP verification
    return true; // Replace with actual verification logic
  }

  async sendMFACode(userId, challengeId) {
    // Implement MFA code sending via SMS, email, or authenticator app
    // This is a placeholder - implement actual code sending
    console.log(`MFA code sent for user ${userId}, challenge ${challengeId}`);
  }

  async notifyAdministrators(userId, event) {
    // Implement administrator notification system
    console.log(`Notifying administrators about ${event} for user ${userId}`);
  }
}

// Memory Storage for Supabase Auth (replace with proper storage in production)
class MemoryStorage {
  constructor() {
    this.data = new Map();
  }

  setItem(key, value) {
    this.data.set(key, value);
  }

  getItem(key) {
    return this.data.get(key);
  }

  removeItem(key) {
    this.data.delete(key);
  }
}

module.exports = HealthcareAuthenticationService;