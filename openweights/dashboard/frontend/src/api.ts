import axios, { AxiosError } from 'axios';
import { Job, Run, Worker, JobWithRuns, RunWithJobAndWorker, WorkerWithRuns, Organization } from './types';
import { supabase } from './supabaseClient';

// In production, use relative paths. In development, use localhost
const API_URL = import.meta.env.PROD ? '' : 'http://localhost:8124';

const getAuthHeaders = async () => {
    const { data: { session } } = await supabase.auth.getSession();
    if (!session?.access_token) {
        throw new Error('No authentication token available');
    }

    return {
        headers: {
            'Authorization': `Bearer ${session.access_token}`,
            'Content-Type': 'application/json'
        }
    };
};

// Helper function to extract error message from API response
const getErrorMessage = (error: unknown): string => {
    if (error instanceof AxiosError && error.response?.data) {
        // FastAPI typically returns error details in the 'detail' field
        if (typeof error.response.data === 'object' && 'detail' in error.response.data) {
            return error.response.data.detail;
        }
        // If the entire response data is a string, use that
        if (typeof error.response.data === 'string') {
            return error.response.data;
        }
    }
    // Fallback to error message or generic error
    return error instanceof Error ? error.message : 'An unknown error occurred';
};

interface CreateOrganizationData {
    name: string;
    secrets: {
        HF_USER: string;
        HF_ORG: string;
        HF_TOKEN: string;
        RUNPOD_API_KEY: string;
    };
}

export const api = {
    // Organizations
    getOrganizations: async () => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.get<Organization[]>(`${API_URL}/organizations/`, config);
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    getOrganization: async (orgId: string) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.get<Organization>(`${API_URL}/organizations/${orgId}`, config);
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    createOrganization: async (data: CreateOrganizationData) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.post<Organization>(`${API_URL}/organizations/`, data, config);
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    updateOrganizationSecrets: async (orgId: string, secrets: Record<string, string>) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.put(
                `${API_URL}/organizations/${orgId}/secrets`,
                secrets,
                config
            );
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    // Jobs
    getJobs: async (orgId: string, status?: string) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.get<Job[]>(`${API_URL}/organizations/${orgId}/jobs/`, {
                ...config,
                params: { status }
            });
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    getJob: async (orgId: string, jobId: string) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.get<JobWithRuns>(`${API_URL}/organizations/${orgId}/jobs/${jobId}`, config);
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    cancelJob: async (orgId: string, jobId: string) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.post<Job>(
                `${API_URL}/organizations/${orgId}/jobs/${jobId}/cancel`,
                {},
                config
            );
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    restartJob: async (orgId: string, jobId: string) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.post<Job>(
                `${API_URL}/organizations/${orgId}/jobs/${jobId}/restart`,
                {},
                config
            );
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    // Runs
    getRuns: async (orgId: string, status?: string) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.get<Run[]>(`${API_URL}/organizations/${orgId}/runs/`, {
                ...config,
                params: { status }
            });
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    getRun: async (orgId: string, runId: string) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.get<RunWithJobAndWorker>(`${API_URL}/organizations/${orgId}/runs/${runId}`, config);
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    getRunLogs: async (orgId: string, runId: string) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.get(`${API_URL}/organizations/${orgId}/runs/${runId}/logs`, {
                ...config,
                responseType: 'text'
            });
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    getRunEvents: async (orgId: string, runId: string) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.get(`${API_URL}/organizations/${orgId}/runs/${runId}/events`, config);
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    // Workers
    getWorkers: async (orgId: string, status?: string) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.get<Worker[]>(`${API_URL}/organizations/${orgId}/workers/`, {
                ...config,
                params: { status }
            });
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    getWorker: async (orgId: string, workerId: string) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.get<WorkerWithRuns>(`${API_URL}/organizations/${orgId}/workers/${workerId}`, config);
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    getWorkerLogs: async (orgId: string, workerId: string) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.get(`${API_URL}/organizations/${orgId}/workers/${workerId}/logs`, {
                ...config,
                responseType: 'text'
            });
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    shutdownWorker: async (orgId: string, workerId: string) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.post<Worker>(
                `${API_URL}/organizations/${orgId}/workers/${workerId}/shutdown`,
                {},
                config
            );
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    // Files
    getFileContent: async (orgId: string, fileId: string) => {
        try {
            const config = await getAuthHeaders();
            console.log('Fetching file content for:', fileId);
            const response = await axios.get(`${API_URL}/organizations/${orgId}/files/${fileId}/content`, {
                ...config,
                responseType: 'text'
            });
            console.log('File content response:', response.data);
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    // Tokens
    createToken: async (orgId: string, name: string, expiresInDays?: number) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.post(`${API_URL}/organizations/${orgId}/tokens`, {
                name,
                expires_in_days: expiresInDays
            }, config);
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    listTokens: async (orgId: string) => {
        try {
            const config = await getAuthHeaders();
            const response = await axios.get(`${API_URL}/organizations/${orgId}/tokens`, config);
            return response.data;
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    },

    deleteToken: async (orgId: string, tokenId: string) => {
        try {
            const config = await getAuthHeaders();
            await axios.delete(`${API_URL}/organizations/${orgId}/tokens/${tokenId}`, config);
        } catch (error) {
            throw new Error(getErrorMessage(error));
        }
    }
};
