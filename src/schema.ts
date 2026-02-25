import { z } from "zod";

export const FitLabelSchema = z.enum(["good_fit", "maybe", "bad_fit"]);

export const GoldenJobSchema = z.object({
  job_id: z.string().min(1),
  title: z.string().min(1),
  company: z.string().min(1),
  location: z.string().min(1).optional(),
  jd_text: z.string().min(40),
  source_url: z.string().url().optional(),
  label: FitLabelSchema,
  score: z.number().int().min(0).max(100),
  reasoning: z.string().min(15).max(600),
});

export type FitLabel = z.infer<typeof FitLabelSchema>;
export type GoldenJob = z.infer<typeof GoldenJobSchema>;

export type JobExportRecord = Record<string, unknown> & {
  id?: string | number;
  job_id?: string | number;
  title?: string;
  company?: string;
  company_name?: string;
  location?: string;
  jd_text?: string;
  description?: string;
  source_url?: string;
  url?: string;
};
