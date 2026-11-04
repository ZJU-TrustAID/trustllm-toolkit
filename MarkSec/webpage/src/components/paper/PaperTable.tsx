import { cn } from "@/lib/cn";

type PaperTableProps = {
  caption: string;
  columns: string[];
  rows: Array<Array<React.ReactNode>>;
  note?: string;
  className?: string;
};

export default function PaperTable({
  caption,
  columns,
  rows,
  note,
  className,
}: PaperTableProps) {
  return (
    <figure className={cn("paper-table-wrap", className)}>
      <figcaption className="paper-table-caption">{caption}</figcaption>
      <div className="overflow-x-auto">
        <table className="paper-table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{column}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {row.map((cell, cellIndex) => (
                  <td key={`${rowIndex}-${cellIndex}`}>{cell}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {note ? <p className="paper-table-note">{note}</p> : null}
    </figure>
  );
}
