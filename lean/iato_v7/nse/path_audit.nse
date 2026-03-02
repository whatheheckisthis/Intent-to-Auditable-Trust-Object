local json = require "json"
local lfs = require "lfs"
local nmap = require "nmap"
local stdnse = require "stdnse"


description = [[
Scaffold NSE script for deterministic host-path metadata and integrity collection.
This script is intentionally bounded for CI proxy worker execution.
]]

author = "IATO-V7"
license = "Same as Nmap--See https://nmap.org/book/man-legal.html"
categories = {"safe", "discovery"}

prerule = function()
  return true
end

local function read_policy(policy_file)
  local fh = io.open(policy_file, "r")
  if not fh then
    return nil, "unable to open policy file"
  end
  local raw = fh:read("*a")
  fh:close()
  local ok, parsed = pcall(json.parse, raw)
  if not ok then
    return nil, "policy json parse error"
  end
  return parsed, nil
end

local function file_mode_string(path)
  local attr = lfs.attributes(path)
  if not attr then
    return nil
  end
  return attr.permissions or "unknown"
end

action = function()
  local root = stdnse.get_script_args("path_audit.root") or "/workspace"
  local policy_file = stdnse.get_script_args("path_audit.policy")
  local schema = stdnse.get_script_args("path_audit.schema") or "1.0.0"
  local release = stdnse.get_script_args("path_audit.release") or "unknown"

  if not policy_file then
    return "path_audit.policy script argument is required"
  end

  local policy, err = read_policy(policy_file)
  if err then
    return string.format("policy load failure: %s", err)
  end

  local output = {
    schema_version = schema,
    release = release,
    root_path = root,
    evaluated = {},
  }

  for _, rule in ipairs(policy.rules or {}) do
    local full_path = string.format("%s/%s", root, rule.path)
    local attr = lfs.attributes(full_path)
    local item = {
      path = full_path,
      exists = attr ~= nil,
      size = attr and attr.size or -1,
      permissions = file_mode_string(full_path),
      expected = rule,
      hash = "scaffold-pending-external-sha256",
      owner = {
        uid = rule.uid or "scaffold",
        gid = rule.gid or "scaffold",
      },
    }
    item.match = item.exists and ((rule.size == nil) or (rule.size == item.size))
    table.insert(output.evaluated, item)
  end

  return stdnse.format_output(true, output)
end
