export interface Database {
  public: {
    Tables: {
      organizations: {
        Row: {
          id: string;
          name: string;
          created_at: string;
        };
        Insert: {
          name: string;
        };
        Update: {
          name?: string;
        };
      };
      organization_members: {
        Row: {
          organization_id: string;
          user_id: string;
          role: 'admin' | 'user';
          email: string;
        };
      };
      organization_secrets: {
        Row: {
          id: string;
          organization_id: string;
          name: string;
          value: string;
          created_at: string;
          updated_at: string;
        };
      };
    };
    Functions: {
      create_organization: {
        Args: { org_name: string };
        Returns: { id: string; name: string };
      };
      invite_organization_member: {
        Args: { org_id: string; member_email: string; member_role: 'admin' | 'user' };
        Returns: { user_id: string; email: string; role: 'admin' | 'user' };
      };
      remove_organization_member: {
        Args: { org_id: string; member_id: string };
        Returns: void;
      };
      update_organization: {
        Args: { org_id: string; new_name: string };
        Returns: void;
      };
      manage_organization_secret: {
        Args: { org_id: string; secret_name: string; secret_value: string };
        Returns: void;
      };
      delete_organization_secret: {
        Args: { org_id: string; secret_name: string };
        Returns: void;
      };
    };
  };
}
