import os
from ai_engine import QueryDecomposer

class JSONToSQLCompiler:
    def __init__(self, json_data):
        self.data = json_data
        # Map task_id to task object for easy lookup
        self.tasks = {t['task_id']: t for t in self.data.get('tasks', [])}

    def compile(self):
        """
        Main entry point. Finds the root task (usually Task 1) and starts compilation.
        """
        if not self.tasks:
            return "-- No tasks found"
        
        root_task_id = self._find_root_task()
        return self._compile_task(root_task_id)

    def _find_root_task(self):
        """
        Identifies the entry point task.
        Logic: The root task is a task that is NOT referenced by any other task 
        in 'structural_logic' (e.g., as a target of a UNION).
        """
        all_ids = set(self.tasks.keys())
        referenced_ids = set()

        for task in self.tasks.values():
            # Check structural_logic for references to other tasks (Set Operations)
            for logic in task.get('structural_logic', []):
                if 'target_task_id' in logic:
                    referenced_ids.add(logic['target_task_id'])
            self._collect_subquery_ids(task, referenced_ids)
        # The roots are the IDs that are present in 'all_ids' but not in 'referenced_ids'
        candidates = list(all_ids - referenced_ids)

        if not candidates:
            # Circular dependency or weird edge case; fallback to the last task defined
            # Error case
            return list(self.tasks.keys())[-1]
        
        # If multiple candidates exist (rare), usually the one with the highest ID 
        # is the aggregator in sequential generation, or we simply pick the first.
        return max(candidates)

    def _collect_subquery_ids(self, obj, refs):
        """Helper to recursively find 'task_id' inside SUBQUERY nodes."""
        if isinstance(obj, dict):
            if obj.get('type') == 'SUBQUERY' and 'target_task_id' in obj:
                refs.add(obj['target_task_id'])
            for v in obj.values():
                self._collect_subquery_ids(v, refs)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_subquery_ids(item, refs)

    def _compile_task(self, task_id):
        """
        Recursively compiles a specific task into SQL.
        """
        task = self.tasks.get(task_id)
        if not task:
            return f"-- Error: Task {task_id} not found"

        # Separate JOINs from SET OPERATIONS (UNION, INTERSECT)
        joins = []
        set_ops = []
        
        for logic in task.get('structural_logic', []):
            if 'JOIN' in logic['type']:
                joins.append(logic)
            elif logic['type'] in ['UNION', 'INTERSECT', 'EXCEPT']:
                set_ops.append(logic)

        # --- Build the Base Query ---
        
        # 1. SELECT (Target)
        select_clause = self._build_select(task.get('target', []), task)

        # 2. FROM
        from_clause = f"FROM {task['main_table']}"

        # 3. JOINs
        join_clause = self._build_joins(joins, task)

        # 4. WHERE
        where_clause = self._build_where(task.get('where_clause'), task)

        # 5. GROUP BY
        group_by_clause = ""
        if task.get('group_by'):
            group_by_clause = "GROUP BY " + ", ".join(task['group_by'])

        # 6. HAVING
        having_clause = self._build_having(task.get('having_clause'), task)

        # 7. ORDER BY
        order_by_clause = self._build_order_by(task.get('order_by'), task)

        # 8. LIMIT
        limit_clause = ""
        if task.get('limit_by'):
            limit_clause = f"LIMIT {task['limit_by']}"

        # Construct the base SQL for this task
        base_parts = [
            select_clause,
            from_clause,
            join_clause,
            where_clause,
            group_by_clause,
            having_clause,
            order_by_clause,
            limit_clause
        ]
        base_sql = "\n".join(part for part in base_parts if part)

        # --- Handle Set Operations (Recursion) ---
        if set_ops:
            final_sql = base_sql
            for op in set_ops:
                target_id = op['target_task_id']
                operator = op['type'] 
                
                # Recursive call to compile the target task
                target_sql = self._compile_task(target_id)
                
                final_sql = f"({final_sql}) \n{operator} \n({target_sql})"
            return final_sql
        
        return base_sql

    def _build_select(self, target_list, current_task):
        """Builds the SELECT clause."""
        columns = []
        # 'is_distinct' is now a property of the Task, not the target dict
        use_distinct = current_task.get('is_distinct', False)

        if not target_list: 
            return "SELECT *"

        for item in target_list:
            val_sql = self._parse_value_node(item['value'], current_task)
            alias = item.get('alias')
            if alias:
                columns.append(f"{val_sql} AS {alias}")
            else:
                columns.append(val_sql)
        
        select_prefix = "SELECT DISTINCT" if use_distinct else "SELECT"
        return f"{select_prefix} {', '.join(columns)}"

    def _build_joins(self, join_logic_list, current_task):
        """Constructs INNER/LEFT JOIN clauses."""
        joins = []
        if not join_logic_list:
            return ""

        for logic in join_logic_list:
            table = logic['table']
            condition = self._parse_condition_node(logic['condition'], current_task)
            joins.append(f"{logic['type']} {table} ON {condition}")
        
        return "\n".join(joins)

    def _build_where(self, where_node, current_task):
        if not where_node:
            return ""
        condition_sql = self._parse_condition_node(where_node, current_task)
        return f"WHERE {condition_sql}"

    def _build_having(self, having_node, current_task):
        if not having_node:
            return ""
        condition_sql = self._parse_condition_node(having_node, current_task)
        return f"HAVING {condition_sql}"

    def _build_order_by(self, order_nodes, current_task):
        if not order_nodes:
            return ""
        orders = []
        for node in order_nodes:
            val = self._parse_value_node(node['value'], current_task)
            direction = node.get('direction', 'ASC')
            orders.append(f"{val} {direction}")
        return "ORDER BY " + ", ".join(orders)

    # --- CORE PARSERS ---

    def _parse_condition_node(self, node, current_task):
        """Recursively parses AND/OR condition trees."""
        if 'logic' in node: 
            logic_op = node['logic']
            sub_conditions = [self._parse_condition_node(c, current_task) for c in node['conditions']]
            return f"({f' {logic_op} '.join(sub_conditions)})"
        
        else:
            operator = node['operator']
            left = self._parse_value_node(node['left'], current_task, operator)
            right = self._parse_value_node(node['right'], current_task, operator)

            # Is the right value a list (semantic search result)
            right_str = str(right).strip().upper()
            is_list_format = "," in str(right) and str(right).strip().startswith("(")
            is_subquery = right_str.startswith("(SELECT") or right_str.startswith("SELECT")
            if is_list_format and is_subquery:
                if operator == "=":
                    operator = "IN"
                elif operator in ["!=", "<>"]:
                    operator = "NOT IN"
            return f"({left} {operator} {right})"

    def _parse_value_node(self, node, current_task, parent_operator=""):
        """Parses atomic values."""
        if not node: return "NULL"
        
        v_type = node.get('type')
        value = node.get('value')

        if v_type == 'LITERAL':
            return f"'{value}'" if isinstance(value, str) else str(value)
        
        elif v_type == 'COLUMN':
            return value
        
        elif v_type == 'FUNCTION':
            func_name = node.get('name', '').upper()
            # Recursive: params are ValueNodes
            params = [self._parse_value_node(p, current_task) for p in node.get('params', [])]
            
            if func_name == "COUNT" and not params:
                return "COUNT(*)"
            return f"{func_name}({', '.join(params)})"

        elif v_type == 'CASE':
            cases = []
            for c in node.get('cases', []):
                cond = self._parse_value_node(c['when'], current_task)
                then_val = self._parse_value_node(c['then'], current_task)
                cases.append(f"WHEN {cond} THEN {then_val}")
            
            else_part = ""
            if 'else' in node:
                else_val = self._parse_value_node(node['else'], current_task)
                else_part = f" ELSE {else_val}"
            
            return f"CASE {' '.join(cases)}{else_part} END"

        elif v_type == 'CONDITION':
            # Value behaves like a boolean/math operation (e.g., amount * 0.2)
            # Delegate to condition parser
            return self._parse_condition_node(value, current_task)

        elif v_type == 'SEMANTIC':
            # Pass operator to decide on Scalar vs List return
            return self._mock_semantic_search(value, node.get('table'), node.get('column'), parent_operator)
        
        elif v_type == 'SUBQUERY':
            # Reference to another task via ID (replaces SUBQUERY)
            ref_task_id = node.get('target_task_id')
            return f"({self._compile_task(ref_task_id)})"

        return "NULL"

    def _mock_semantic_search(self, search_term, table, column, operator):        
        """Simulates Vector DB retrieval."""
        term = str(search_term).lower()
        op = str(operator).upper().strip()
        set_compatible_ops = ["IN", "NOT IN", "=", "!=", "<>"]
        force_single_value = op not in set_compatible_ops
        if "premium" in term:
            return "('Premium', 'VIP', 'Gold')"
        elif "rich" in term:
            return "('Prague', 'Brno')" 
        elif "female" in term:
            return "'F'"
        elif "issuance after" in term:
            return "'ISSUANCE_AFTER_TRANS'"
        return f"'{search_term}'"

if __name__ == "__main__":
    if not os.path.exists('info/database_info.json'):
        print("[ERROR] database_info.json not found.")
        exit()

    try:
        decomposer = QueryDecomposer(info_path='info/database_info.json')
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        exit()

    test_queries = [
        "Show me the average balance of female clients from Prague who have gold cards.", # Complex join
        "List all loans in Prague and find the youngest client in Brno.", # Independent Multi-Task
        "Who are the clients with a balance higher than the average balance?", # Subquery
        "Order pizza for me.", # Should trigger is_achievable: False,
        "List the IDs of clients who live in Prague and union them with the IDs of clients who have a gold card.", # Union,
        "Name the account numbers of female clients who are oldest and have lowest average salary?", # moderate
        "List out the accounts who have the earliest trading date in 1995 ?", # simple
        "For the branch which located in the south Bohemia with biggest number of inhabitants, what is the percentage of the male clients?" # challenging
        "Name the account numbers of female clients who are oldest and have lowest average salary?" # moderate
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"QUERY {i}: {query}")
        
        # Step 1: Decompose
        json_result, total_tokens =  decomposer.decompose_query("financial", query)

        tasks = json_result.get("tasks", [])
        if not tasks:
            print("[INFO] No tasks returned.")
            continue
            
        if not tasks[0].get("is_achievable", True):
            print(f"[INFO] Query unachievable: {tasks[0].get('error')}")
            continue

        # Step 2: Compile
        compiler = JSONToSQLCompiler(json_result)
        sql_output = compiler.compile()

        print("GENERATED SQL:")
        print(sql_output)
        print("-" * 50)