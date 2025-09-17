export interface Organization {
    id: string;
    name: string;
    created_at: string;
}

export interface Job {
    id: string;
    type: string;
    status: string;
    model?: string;
    script?: string;
    params?: any;
    outputs?: any;
    requires_vram_gb?: number;
    docker_image?: string;
    created_at: string;
}

export interface Run {
    id: string;
    job_id: string;
    worker_id?: string;
    status: string;
    log_file?: string;
    created_at: string;
}

export interface Worker {
    id: string;
    status: string;
    gpu_type?: string;
    gpu_count?: number;
    vram_gb?: number;
    docker_image?: string;
    cached_models?: string[];
    pod_id?: string;
    ping?: string;
    created_at: string;
}

export interface JobWithRuns extends Job {
    runs: Run[];
}

export interface RunWithJobAndWorker extends Run {
    job: Job;
    worker?: Worker;
}

export interface WorkerWithRuns extends Worker {
    runs: Run[];
}

export interface FileContent {
    id: string;
    content: string;
    loading: boolean;
    error?: string;
}

export type FileContents = {
    [key: string]: FileContent;
};
