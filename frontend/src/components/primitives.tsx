import type { ReactNode } from "react";
import { AlertTriangle, Loader2, Inbox } from "lucide-react";
import type { StatusKind } from "../lib/status";
import { STATUS_LABEL } from "../lib/status";

// ---- Mono label ----
export function MonoLabel({ children }: { children: ReactNode }) {
  return <span className="mono-label">{children}</span>;
}

// ---- Pill ----
export function Pill({
  children,
  variant,
  dot,
}: {
  children: ReactNode;
  variant?: "accent" | StatusKind;
  dot?: boolean;
}) {
  const cls = variant ? `pill pill--${variant}` : "pill";
  return (
    <span className={cls}>
      {dot && <span className="pill__dot" aria-hidden />}
      {children}
    </span>
  );
}

export function StatusPill({ status }: { status: StatusKind }) {
  return (
    <Pill variant={status} dot>
      {STATUS_LABEL[status]}
    </Pill>
  );
}

// ---- Card ----
export function BentoCard({
  children,
  className = "",
  major = false,
  interactive = false,
  as: Tag = "section",
  ...rest
}: {
  children: ReactNode;
  className?: string;
  major?: boolean;
  interactive?: boolean;
  as?: keyof JSX.IntrinsicElements;
} & Record<string, unknown>) {
  const cls = [
    "card",
    major ? "card--major" : "",
    interactive ? "card--interactive" : "",
    className,
  ]
    .filter(Boolean)
    .join(" ");
  return (
    <Tag className={cls} {...rest}>
      {children}
    </Tag>
  );
}

// ---- Metric card ----
export function MetricCard({
  label,
  value,
  sub,
  status,
  interactive = true,
}: {
  label: string;
  value: ReactNode;
  sub?: ReactNode;
  status?: StatusKind;
  interactive?: boolean;
}) {
  return (
    <BentoCard interactive={interactive}>
      <div className="card__head">
        <MonoLabel>{label}</MonoLabel>
        {status && <StatusPill status={status} />}
      </div>
      <div className="metric__value">{value}</div>
      {sub && <div className="metric__sub">{sub}</div>}
    </BentoCard>
  );
}

// ---- Button ----
export function Button({
  children,
  variant = "default",
  onClick,
  disabled,
  type = "button",
  title,
  ariaLabel,
}: {
  children: ReactNode;
  variant?: "default" | "primary" | "danger" | "ghost";
  onClick?: () => void;
  disabled?: boolean;
  type?: "button" | "submit";
  title?: string;
  ariaLabel?: string;
}) {
  const cls = variant === "default" ? "btn" : `btn btn--${variant}`;
  return (
    <button
      type={type}
      className={cls}
      onClick={onClick}
      disabled={disabled}
      title={title}
      aria-label={ariaLabel}
    >
      {children}
    </button>
  );
}

// ---- States ----
export function LoadingState({ label = "Loading" }: { label?: string }) {
  return (
    <div className="stack stack-3 center" style={{ padding: "var(--space-6)", alignItems: "center" }}>
      <Loader2 size={22} className="spin tertiary" aria-hidden />
      <MonoLabel>{label}</MonoLabel>
    </div>
  );
}

export function EmptyState({ title, hint }: { title: string; hint?: string }) {
  return (
    <div className="stack stack-3 center" style={{ padding: "var(--space-6)", alignItems: "center" }}>
      <Inbox size={22} className="tertiary" aria-hidden />
      <h3>{title}</h3>
      {hint && <p className="tertiary" style={{ maxWidth: 360 }}>{hint}</p>}
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  return (
    <div
      className="stack stack-3"
      role="alert"
      style={{ padding: "var(--space-5)", alignItems: "flex-start" }}
    >
      <div className="row row-3" style={{ color: "var(--status-drift)" }}>
        <AlertTriangle size={18} aria-hidden />
        <MonoLabel>Unavailable</MonoLabel>
      </div>
      <p className="muted" style={{ fontSize: "0.9rem" }}>
        {message}
      </p>
      <p className="tertiary" style={{ fontSize: "0.8rem" }}>
        This is shown as unavailable, not as backend data.
      </p>
    </div>
  );
}
